"""Whole image reader and RNA-Seq reader.

@authors: Francisco Carrillo-Perez
"""
import data_reader
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import pandas as pd
import random
import h5py
from PIL import Image


class ImageGenReader(data_reader.Data_reader):
    """Img reader for full size WSI images."""
    def __init__(self, folder_name=['im_resized', 'gen_mrmr'], np_shape=(299,299), formats=['png', '.csv']):
        #data_reader.Data_reader.__init__(self, folder_name, np_shape, formats)
        super(ImageGenReader, self).__init__(folder_name, np_shape, formats)
        self.data = {}
        self.data['train'] = {'x': [], 'y': []}
        self.data['val'] = {'x': [], 'y': []}
        self.data['test'] = {'x': [], 'y': []}
        self.case_ids = {}
        self.case_ids['train'] = []
        self.case_ids['test'] = []
        self.case_ids['val'] = []

    def __del__(self):
        super(ImageGenReader, self).__del__()

    def read_data(self, ohe, paths, dataset='train', stack=True, val_pat=False):
        squa = ['Basaloid squamous cell carcinoma',
        'Papillary squamous cell carcinoma',
        'Squamous cell carcinoma, NOS',
        'Squamous cell carcinoma, keratinizing, NOS',
        'Squamous cell carcinoma, large cell, nonkeratinizing, NOS',
        'Squamous cell carcinoma, small cell, nonkeratinizing']
        adeno = ['Adenocarcinoma with mixed subtypes', 'Adenocarcinoma, NOS', 'Bronchiolo-alveolar adenocarcinoma, NOS',
                'Bronchiolo-alveolar carcinoma, non-mucinous', 'Clear cell adenocarcinoma, NOS', 'Micropapillary carcinoma, NOS',
            'Papillary adenocarcinoma, NOS', 'Solid carcinoma, NOS', 'Bronchio-alveolar carcinoma, mucinous']
        for path in tqdm(paths):
            case_id = os.path.split(path)[-1]
            data_path = os.path.join(path, self.folder_name[0])
            data_gen_path = os.path.join(path, self.folder_name[1])
            type = os.path.split(path)[-1].split('-')[-1]
            if type[0] == '1':
                label = ['healthy']
            elif type[0] == '0':
                cli_path = os.path.join(path, 'clinical')
                filename = glob.glob(cli_path+'/*.csv')
                try:
                    data = pd.read_csv(filename[0])
                except:
                    pass
                try:
                    label = data['primary_diagnosis'].values
                except:
                    print("error with label obtention for {}".format(path))
                    continue
                if label in squa:
                    label = ['squamous']
                    #label = ['tumor']
                elif label in adeno:
                    label = ['adenocarcinoma']
                    #label = ['tumor']

            if (not os.path.exists(data_path) or len(os.listdir(data_path)) == 0 or 
                not os.path.exists(data_gen_path) or len(os.listdir(data_gen_path)) == 0):
                pass
                #self.data[dataset][case_id] = [np.empty(self.np_shape) * np.nan]
            else:
                if dataset == 'test' or val_pat:
                    list_case_ids = []
                
                gen_file = glob.glob(data_gen_path + "/*" + self.formats[1])[0]
                for file in glob.glob(data_path + "/*" + self.formats[0]):
                    if dataset == 'test' or val_pat:
                        list_case_ids.append(file)
                    else:
                        self.data[dataset]['x'] += [file]
                    
                    if dataset != 'test' and not val_pat:
                        self.data[dataset]['y'] += [label]
                
                self.case_ids[dataset] += [case_id]
                    
                if dataset == 'test' or val_pat:
                    self.data[dataset]['y'] += [label]
                 
                if dataset == 'test' or val_pat:
                     self.data[dataset]['x'] += [(list_case_ids, gen_file)]
        
        #self.data[dataset]['x'] = np.asarray(self.data[dataset]['x'])
        self.data[dataset]['y'] = np.asarray(self.data[dataset]['y'])
        self.data[dataset]['y'] = ohe.transform(self.data[dataset]['y'].reshape(-1,1))