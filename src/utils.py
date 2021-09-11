import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from tqdm import tqdm
from glob import glob

# Plot function
def plot_results(results_acc, results_loss, save_path, prefix):
    epochs = range(len(results_acc['train']))

    # Plotting accuracy
    plt.figure()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, results_acc['train'], label='train')
    plt.plot(epochs, results_acc['val'], label='val')
    plt.legend()
    plt.savefig(save_path + prefix + '_accuracy.png')
    plt.close()

    # Plotting loss
    plt.figure()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, results_loss['train'], label='train')
    plt.plot(epochs, results_loss['val'], label='val')
    plt.legend()
    plt.savefig(save_path + prefix + '_loss.png')
    plt.close()


def save_preds(preds, labels, case_ids, save_path, prefix):
    data = pd.DataFrame()
    data['Case Ids'] = case_ids
    data['Pred'] = preds
    data['Real'] = labels

    data.to_csv(save_path + prefix + '_predictions.csv', index=False, sep=',')

def get_paths(splits_folder, db_path, split, cnn=False, multitest=False):
    random.seed(27*split)
    paths = {}
    f = open('../'+ splits_folder +'/train_'+str(split) + '.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['train'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    f = open('../'+ splits_folder + '/val_'+str(split)+'.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['test'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    '''
    if multitest:
        f = open('../' + splits_folder +'/test_'+str(split)+'.txt', 'r')
    else:
        f = open('../' + splits_folder +'/test.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['test'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    '''
    if cnn:
        val_len = int(len(paths['train'])*0.9)
        random.shuffle(paths['train'])
        paths['val'] = paths['train'][val_len:]
        paths['train'] = paths['train'][0:val_len]

    return paths

def get_paths_gen(splits_folder, db_path, split, cnn=False, multitest=False):
    random.seed(27*split)
    paths = {}
    f = open('../'+ splits_folder +'/train_'+str(split) + '.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['train'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    f = open('../'+ splits_folder + '/val_'+str(split)+'.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['val'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    
    if multitest:
        f = open('../' + splits_folder +'/test_'+str(split)+'.txt', 'r')
    else:
        f = open('../' + splits_folder +'/test.txt', 'r')
    aux = [os.path.join(db_path, patient_id) for patient_id in f.read().splitlines()]
    paths['test'] = [os.path.join(patient_id, case_id) for patient_id in aux for case_id in os.listdir(patient_id)]
    f.close()
    
    return paths

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def maxpool_1d(x, sections=32):
    features = []
    for arr in np.split(x, sections):
        features.append(np.max(arr))
    
    return np.asarray(features)

def get_best_n_genes(X, y, max_genes=11):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.svm import SVC
    X, y = np.asarray(X), np.asarray(y)
    kf = KFold(n_splits=5)
    results_accs = []
    for n_genes in tqdm(range(1, max_genes)):
        accs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index,0:n_genes], X[test_index,0:n_genes]
            y_train, y_test = y[train_index], y[test_index]

            scaler = MinMaxScaler(feature_range=(-1,1))
            
            #tuned_parameters = [{'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}]
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-7, 2**-5, 2**-2, 2, 2**4, 2**7],
                                'C': [2**-7, 2**-5, 2**-2, 2, 2**4, 2**7]}]
            clf = GridSearchCV(
                            SVC(probability=True), tuned_parameters, scoring='accuracy'
                        )
            
            x_train_new = scaler.fit_transform(X_train)
            clf.fit(x_train_new, y_train.argmax(axis=1))
           
            x_test = scaler.transform(X_test)
            test_preds = clf.predict(x_test)
            
            corrects = np.sum(test_preds == y_test.argmax(axis=1))
            test_acc = (corrects / x_test.shape[0]) * 100
           
            accs.append(test_acc)
        
        results_accs.append(np.mean(accs))
    maximum = max(results_accs)
    return results_accs.index(maximum) + 1


# stadio, join_stadio not used, but keeped to maintain compatibility
def get_LC_inputs(mode, paths, ohe, dataReaders, verbose=False):
    """Helper function to get inputs."""
    data_x = {}
    data_y = {}
    final_paths = {}
    # Get data
    for r_name in dataReaders.keys():
        print('Reading data from ' + str(r_name))
        subnet_data_x = []
        subnet_final_paths = []
        subnet_data_y = []
        # Get data
        for path in paths:
            type_ = os.path.split(path)[-1].split('-')[-1]
            if type_[0] == '1':
                label = ['healthy']
            elif type_[0] == '0':
                cli_path = os.path.join(path, 'clinical')
                filename = glob(cli_path+'/*.csv')
                try:
                    data = pd.read_csv(filename[0])
                except:
                    pass
                try:
                    project_id = data['project_id'].values[0]
                except:
                    print("error with label obtention for {}".format(path))
                    continue
    
                if project_id == 'TCGA-LUSC':
                    label = ['squamous']
                elif project_id == 'TCGA-LUAD':
                    label = ['adenocarcinoma']
                else:
                    continue
            subnet_final_paths.append(path)
            subnet_data_y += label
            case_id = os.path.split(path)[-1]
            if dataReaders[r_name].get_data(mode, case_id).shape[0] == 0:
                print(case_id)
            subnet_data_x += [dataReaders[r_name].get_data(mode, case_id)]

        if subnet_data_x == []:
            print("Empty")
            return None, None
        else:
            data_x[r_name] = np.stack(subnet_data_x, axis=0)

        data_y[r_name] = np.asarray(subnet_data_y)
        data_y[r_name] = ohe.transform(data_y[r_name].reshape(-1, 1))
        final_paths[r_name] = subnet_final_paths
    
    return data_x, data_y, final_paths


def index_to_keep(array, condition):
    # condition -> np.full(6,-3)
    return np.unique(np.where(array != condition)[0])