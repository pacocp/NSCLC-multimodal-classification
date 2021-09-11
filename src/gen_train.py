import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from gen_reader import GenReader
from utils import get_paths

# Parsing arguments by config file
parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--c', action="store", dest="config", type=str,
                    default='config_default.json')
args = parser.parse_args()
config_name = args.config
with open(config_name, 'r') as f:
    config = json.load(f)

# Config information
splits_folder = config['splits_path']
splits = config['n_splits']
batch_size = config['batch_size']
lr = config['lr']
model_name = config['model_name']
epochs = config['epochs']
bool_lr_scheduler = config['lr_scheduler']
results_folder_name = config['results_folder_name']

# Global parameters
classes = np.array(['healthy', 'adenocarcinoma', 'squamous'])
healthy = True

db_path = '../LungCancer-DB'

ohe = preprocessing.OneHotEncoder(sparse=False)
ohe.fit(classes.reshape(-1,1))

# Creating folder for saving results
if not os.path.isdir('results/'+results_folder_name):
    os.mkdir('results/'+results_folder_name)

test_accs = []
val_accs = []
train_accs = []

for n_genes in [3,6,10]:
    print(n_genes)
    # Creating folder for saving results
    if not os.path.isdir('results/'+results_folder_name+'/'+str(n_genes)):
        os.mkdir('results/'+results_folder_name+'/'+str(n_genes))
    save_path = 'results/'+results_folder_name+'/'+str(n_genes) +'/'
    execution_log = open('execution_'+model_name+'_genes.txt', 'w+')
    for i in range(splits):
        print('Split {}/{}'.format(i, splits))

        """ Create readers """
        dataReaders = {}
        dataReaders['Gen'] = GenReader(folder_name='gen_mrmr_good_split'+str(2), np_shape=(n_genes), num_variables=n_genes, formats=['csv'])                                                
        """ Get paths """
        # Dataset paths
        datasets = ['test', 'train']
        
        paths = get_paths(splits_folder, db_path, i, multitest=False)

        """ Read data """
        
        for key in dataReaders:
            print('Read data ({})'.format(key))
            for dataset in datasets:
                dataReaders[key].read_data(paths=paths[dataset], ohe=ohe, dataset=dataset, val_pat=False)
         
        # getting datasets
        x_train = dataReaders['Gen'].data['train']['x']
        y_train = dataReaders['Gen'].data['train']['y']

        x_train = np.asarray(x_train)

        x_test = dataReaders['Gen'].data['test']['x']
        y_test = dataReaders['Gen'].data['test']['y']
        x_test = np.asarray(x_test)

        scaler = MinMaxScaler(feature_range=(-1,1))
        
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-7, 2**-5, 2**-2, 2, 2**4, 2**7],
                            'C': [2**-7, 2**-5, 2**-2, 2, 2**4, 2**7]}]
        clf = GridSearchCV(
                        SVC(probability=True), tuned_parameters, scoring='accuracy'
                    )
        
        x_train_new = scaler.fit_transform(x_train)
        print(x_train_new.shape)
        clf.fit(x_train_new, y_train.argmax(axis=1))
        print(clf.best_params_)
        best_params = clf.best_params_
        train_preds = clf.predict(x_train_new)
        train_probs = clf.predict_proba(x_train_new)
        corrects = np.sum(train_preds == y_train.argmax(axis=1))
        train_acc = (corrects / x_train_new.shape[0]) * 100
        print('kNN train acc: {}'.format(train_acc))
        train_accs.append(train_acc)

        x_test = scaler.transform(x_test)
        test_preds = clf.predict(x_test)
        test_probs = clf.predict_proba(x_test)
        corrects = np.sum(test_preds == y_test.argmax(axis=1))
        test_acc = (corrects / x_test.shape[0]) * 100
        print('SVM test acc: {}'.format(test_acc))
        test_accs.append(test_acc)

        print("Saving SVM predictions... \n")

        data = pd.DataFrame()
        data['Case_Ids'] = dataReaders['Gen'].case_ids['test']
        data['Preds'] = test_preds
        data['Prob LUAD'] = test_probs[:, 0]
        data['Prob HLT'] = test_probs[:, 1]
        data['Prob LUSC'] = test_probs[:, 2]
        data['Real'] = y_test.argmax(axis=1)
        data.to_excel(save_path+model_name+'_split'+str(i)+'_'+str(n_genes)+'gen_test_mrmr.xlsx')

        data = pd.DataFrame()
        data['Case_Ids'] = dataReaders['Gen'].case_ids['train']
        data['Preds'] = train_preds
        data['Prob LUAD'] = train_probs[:, 0]
        data['Prob HLT'] = train_probs[:, 1]
        data['Prob LUSC'] = train_probs[:, 2]
        data['Real'] = y_train.argmax(axis=1)
        data.to_excel(save_path+model_name+'_split'+str(i)+'_'+str(n_genes)+'gen_train_mrmr.xlsx')

    print("RNA-Seq Train: {} +- {}".format(np.mean(train_accs), np.std(train_accs)))
    execution_log.write("RNA-Seq Train: {} +- {}".format(np.mean(train_accs), np.std(train_accs)))
    execution_log.write('\n')
    print("RNA-Seq Test: {} +- {}".format(np.mean(test_accs), np.std(test_accs)))
    execution_log.write("RNA-Seq Test: {} +- {}".format(np.mean(test_accs), np.std(test_accs)))
    execution_log.write('\n')
            
    execution_log.close()
