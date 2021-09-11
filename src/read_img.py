import data_reader
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import pandas as pd
import random
from PIL import Image


class ImageReader(data_reader.Data_reader):
    """Img reader for full size WSI images."""
    def __init__(self, folder_name='im_resized', np_shape=(299,299), formats=['png'], patch_size=126,
                 max_patches=None):
        super(ImageReader, self).__init__(folder_name, np_shape, formats)
        self.patch_size = patch_size
        self.data = {}
        self.data['train'] = {'x': [], 'y': []}
        self.data['val'] = {'x': [], 'y': []}
        self.data['test'] = {'x': [], 'y': []}
        self.case_ids = []
        self.max_patches = max_patches

    def __del__(self):
        super(ImageReader, self).__del__()

    def read_file(self, file):
        im = cv2.imread(file).astype(np.uint8)
        nobgr_img_blocks = []
        for j in range(0,self.np_shape[1],self.patch_size):
            block_j = im[:, j:j+self.patch_size]
            for i in range(0,self.np_shape[0],self.patch_size):
                block = block_j[i:i+self.patch_size]
                
                number_of_whites = 0
                for i in range(3):
                    if(((block[:][:][i] >= 220).sum() == block[:][:][i].shape[0]).astype(np.int)):
                        number_of_whites += 1
                if(number_of_whites < 2):
                    nobgr_img_blocks.append(block)
        # selecting a maximum number of patches
        if self.max_patches != None and len(nobgr_img_blocks) > self.max_patches:
            indexes = random.sample(range(len(nobgr_img_blocks)), self.max_patches)
            nobgr_blocks = nobgr_img_blocks[indexes]
        else:
            nobgr_blocks = nobgr_img_blocks
        return nobgr_blocks
    
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
            data_path = os.path.join(path, self.folder_name)
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
                elif label in adeno:
                    label = ['adenocarcinoma']
            
            if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
                pass
            else:
                if dataset == 'test' or val_pat:
                    list_case_ids = []
                for format in self.formats:
                    for file in glob.glob(data_path + "/*" + format):
                        if dataset == 'test' or val_pat:
                            list_case_ids.append(file)
                        else:
                            self.data[dataset]['x'] += [file]
                        
                        if dataset != 'test' and not val_pat:
                            self.data[dataset]['y'] += [label]
                        self.case_ids += [case_id]
                     
                    if dataset == 'test' or val_pat:
                        self.data[dataset]['y'] += [label]
                 
                if dataset == 'test' or val_pat:
                     self.data[dataset]['x'] += [list_case_ids]
        
        self.data[dataset]['x'] = np.asarray(self.data[dataset]['x'])
        self.data[dataset]['y'] = np.asarray(self.data[dataset]['y'])
        self.data[dataset]['y'] = ohe.transform(self.data[dataset]['y'].reshape(-1,1))