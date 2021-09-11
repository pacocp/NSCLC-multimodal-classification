import data_reader as dr
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob

class GenReader(dr.Data_reader):
    """RNA-Seq counts file reader."""
    def __init__(self, folder_name='gen', np_shape=(10), num_variables=4, formats=['csv']):
        dr.Data_reader.__init__(self, folder_name, np_shape, formats)
        self.num_variables = num_variables
        self.data['train'] = {'x': [], 'y': []}
        self.data['val'] = {'x': [], 'y': []}
        self.data['test'] = {'x': [], 'y': []}
        self.case_ids = {}
        self.case_ids['train'] = []
        self.case_ids['test'] = []
        self.case_ids['val'] = []

    def read_file(self, file, drop_last=False):
        data = pd.read_csv(file, sep=',')
        # drop last column
        if(drop_last):
            data = data.iloc[:, :-1]
        # removing first column
        try:
            values = data.values[0, 1:self.num_variables+1].astype('float32')
        except:
            values = data.values[0, 2:self.num_variables+2].astype('float32')
        
        return values

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
            data_gen_path = os.path.join(path, self.folder_name)
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

            if (not os.path.exists(data_gen_path) or len(os.listdir(data_gen_path)) == 0):
                pass
            else:
                
                for file in glob.glob(data_gen_path + "/*" + self.formats[0]):
                    self.data[dataset]['x'] += [self.read_file(file)]
                    self.data[dataset]['y'] += [label]
                
                self.case_ids[dataset] += [case_id]
                    
        
        self.data[dataset]['y'] = np.asarray(self.data[dataset]['y'])
        self.data[dataset]['y'] = ohe.transform(self.data[dataset]['y'].reshape(-1,1))