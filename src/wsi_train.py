import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import Dataset, TestDataset
from read_img import ImageReader
from train_pred import predict_WSI, train_model
from utils import get_paths, init_weights

torch.multiprocessing.set_sharing_strategy('file_system')

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
device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

db_path = '../LungCancer-DB'

ohe = preprocessing.OneHotEncoder(sparse=False)
ohe.fit(classes.reshape(-1,1))

# Creating folder for saving weights
if not os.path.isdir('weights/'+results_folder_name):
    os.mkdir('weights/'+results_folder_name)

# Creating folder for saving results
if not os.path.isdir('results/'+results_folder_name):
    os.mkdir('results/'+results_folder_name)

for i in range(5,splits):
    print('Split {}/{}'.format(i, splits))

    """ Create readers """
    dataReaders = {}
    dataReaders['CNN'] = ImageReader(folder_name='img_patches', np_shape=(897, 897, 3), 
                                     formats=['.jpeg'], patch_size=512)
    
    """ Get paths """
    # Dataset paths
    datasets = ['train', 'val','test']
    
    paths = get_paths(splits_folder, db_path, i, cnn=True, multitest=False)


    """ Read data """
    for key in dataReaders:
        print('Read data ({})'.format(key))
        for dataset in datasets:
            dataReaders[key].read_data(paths=paths[dataset], ohe=ohe, dataset=dataset)

    # Shuffle train set
    index_train = np.random.randint(0, len(dataReaders['CNN'].data['train']['x']), len(dataReaders['CNN'].data['train']['x']))
    dataReaders['CNN'].data['train']['x'] = dataReaders['CNN'].data['train']['x'][index_train]
    dataReaders['CNN'].data['train']['y'] = dataReaders['CNN'].data['train']['y'][index_train]
    
    # create train dataloader
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset_train = Dataset(dataReaders['CNN'].data['train']['x'], 
                    dataReaders['CNN'].data['train']['y'], train_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=32,
                                  shuffle=True, num_workers=8, 
                                  pin_memory=True)

    dataset_val = Dataset(dataReaders['CNN'].data['val']['x'], 
                    dataReaders['CNN'].data['val']['y'], val_transform)

    dataloader_val = DataLoader(dataset_val, batch_size=32,
                                shuffle=False, num_workers=8, 
                                pin_memory=True)

    dataset_test = TestDataset(dataReaders['CNN'].data['test']['x'],
                      dataReaders['CNN'].data['test']['y'], val_transform)
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 pin_memory=True)
    
    dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
    dataset_sizes = {x: len(dataReaders['CNN'].data[x]['x']) for x in ['train', 'val', 'test']}
    
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
   
    model.fc =  nn.Sequential(
                  nn.Dropout(p=0.5),  
                  nn.Linear(num_ftrs, len(classes))
                )

    # initializing weights with xavier
    model.fc.apply(init_weights)
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # saving results to path
    save_path = 'results/'+results_folder_name+'/'

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    if bool_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    results = train_model(model=model, criterion=criterion, optimizer=optimizer, dataloaders=dataloaders, 
                dataset_sizes=dataset_sizes, lr_scheduler=lr_scheduler, save_path=save_path, num_epochs=epochs, 
                name=str(i), verbose=True)

    print('Best acc in split {}: {}'.format(i, results['best_acc']))

    print("Saving model's weights to folder")
    torch.save(results['model'].state_dict(), 'weights/'+results_folder_name+'/final_weights_split'+str(i)+'.pkl')

    # write dict results on file
    a_file = open(save_path+results_folder_name+'_split'+str(i)+'_results_Epoch_'+str(results['best_epoch'])+'.pkl', "wb")
    pickle.dump(results, a_file)
    a_file.close()

    model = results['model']
    
    model.eval()
    # save val and train preds

    # train

    data = pd.DataFrame()
    data['Case_Ids'] = results['train_case_ids'][0]
    data['Preds'] = results['train_preds']
    data['Real'] = results['train_labels']
    data.to_excel(save_path+results_folder_name+'_split'+str(i)+'_train_Epoch'+str(results['best_epoch'])+'.xlsx')
    
    # val
    val_case_ids = dataReaders['CNN'].data['val']['x']
    val_labels = np.argmax(dataReaders['CNN'].data['val']['y'], axis=1)

    data = pd.DataFrame()
    data['Case_Ids'] = val_case_ids
    data['Preds'] = results['val_preds']
    data['Real'] = val_labels
    data.to_excel(save_path+results_folder_name+'_split'+str(i)+'_val_Epoch'+str(results['best_epoch'])+'.xlsx')
    
    # predict on test set  
    test_results = predict_WSI(model, dataloader_test, dataset_sizes['test'])
    print('Test acc: {:.4f}\n'.format(test_results['acc']))

    # write dict results on file
    a_file = open(save_path+results_folder_name+'_split'+str(i)+'_test_results.pkl', "wb")
    pickle.dump(test_results, a_file)
    a_file.close()

    test_labels = np.argmax(dataReaders['CNN'].data['test']['y'], axis=1)
    case_ids_test = [x[0].split('/')[3] for x in dataReaders['CNN'].data['test']['x']]

    data = pd.DataFrame()
    data['Case_Ids'] = case_ids_test
    data['Preds'] = test_results['preds']
    data['Prob LUAD'] = test_results['probs']['LUAD']
    data['Prob HLT'] = test_results['probs']['HLT']
    data['Prob LUSC'] = test_results['probs']['LUSC']
    data['Real'] = test_results['global_labels']
    data.to_excel(save_path+results_folder_name+'_split'+str(i)+'_test.xlsx')

    paths = get_paths(splits_folder, db_path, i, cnn=False, multitest=False)
    
    # TRAIN SLIDE
    dataReaders['CNN_slide'] = ImageReader(folder_name='img_patches', np_shape=(897, 897, 3), 
                                     formats=['.jpeg'], patch_size=512)
    dataReaders['CNN_slide'].read_data(paths=paths['train'], ohe=ohe, dataset='train', val_pat=True)

    dataset_train_slide = TestDataset(dataReaders['CNN_slide'].data['train']['x'],
                      dataReaders['CNN_slide'].data['train']['y'], val_transform)
    dataloader_train_slide = DataLoader(dataset_train_slide, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 pin_memory=True)
    
    train_size = len(dataReaders['CNN_slide'].data['train']['x'])
    train_results = predict_WSI(model, dataloader_train_slide, train_size)

    # write dict results on file
    a_file = open(save_path+results_folder_name+'_split'+str(i)+'_train_slide_results.pkl', "wb")
    pickle.dump(train_results, a_file)
    a_file.close()

    train_labels = np.argmax(dataReaders['CNN_slide'].data['train']['y'], axis=1)
    case_ids_train = [x[0].split('/')[3] for x in dataReaders['CNN_slide'].data['train']['x']]

    data = pd.DataFrame()
    data['Case_Ids'] = case_ids_train
    data['Preds'] = train_results['preds']
    data['Prob LUAD'] = train_results['probs']['LUAD']
    data['Prob HLT'] = train_results['probs']['HLT']
    data['Prob LUSC'] = train_results['probs']['LUSC']
    data['Real'] = train_results['global_labels']
    data.to_excel(save_path+results_folder_name+'_split'+str(i)+'_train_slide.xlsx')
