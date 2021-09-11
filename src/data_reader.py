import numpy as np
import glob
import os
import random
from tqdm import tqdm
import gc


def memory(n):
    import psutil
    import os
    process = psutil.Process(os.getpid())
    print(str(n) + " : " + str(process.memory_info()[0]/1024/1024/1024))
    print(str(n) + " : " + str(psutil.virtual_memory()))


def mem_obj(obj):
    import sys
    print(str(sys.getsizeof(obj)/1024/1024/1024))


class Data_reader():
    # Override by subclasses

    def __init__(self, folder_name, np_shape, formats):
        self.folder_name = folder_name
        self.np_shape = np_shape
        self.formats = formats
        self.data = {}
        self.data['train'] = {}
        self.data['val'] = {}
        self.data['test'] = {}

    def __del__(self):
        del self.data

    # Search for all folder with name folder_name within the paths passed
    # Read all files inside using the read_file method and create a list
    # Finally, all file data is added to the data dictionary
    def read_data(self, paths, dataset='train'):
        for path in tqdm(paths):
            case_id = os.path.split(path)[-1]
            data_path = os.path.join(path, self.folder_name)
            if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
                #pass
                self.data[dataset][case_id] = [np.full(self.np_shape,-3)]
            else:
                file_data = []
                for format in self.formats:
                    for file in glob.glob(data_path + "/*" + format):
                        file_data += [self.read_file(file)]
                self.data[dataset][case_id] = file_data

    def read_file(self, file):
        print("Read method not overwritten!")
        return None

    # Return a random data value for a case (if there are multiple)

    def get_data(self, mode, case_id):
        input_data = None
        if mode in self.data and case_id in self.data[mode]:
            input_data = self.data[mode][case_id]
            input_data = input_data[np.random.randint(0, len(input_data))]
        else:
            print("Error, {} or {} not present in data, returning None".format(mode, case_id))
        return input_data
