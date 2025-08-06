import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import re
from scipy.optimize import curve_fit
import numpy as np


def quadratic_curve(x, a):
    return a * x**2

class MGDataset(Dataset):
    def __init__(self, config=None, data_dir=None, mode='train'):
        self.data_dir = data_dir
        self.config = config
        self.mode = mode
        self.read_files(self.data_dir)
        
        super().__init__()
        
    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, idx):
        return self.dict_list[idx]
    
    def read_files(self, data_dir):
        self.dict_list = []
        dirs = os.listdir(data_dir)
        
        for f in dirs:
            ff_list = os.listdir(os.path.join(self.data_dir, f))
            ff_list = [i for i in ff_list if i.endswith('.txt')]
            if self.mode == 'train':
                ff_list = ff_list[:-1] #Take every element except the last one
            else:
                ff_list = ff_list[-1:] #Take only the last element
            other_input = self.process_folder_name(f)

            for j in ff_list:
                txt_path = os.path.join(self.data_dir, f, j)
                mt_val = self.process_file_name(j)
                temp_result = {}
                data = np.array(self.read_txt(txt_path))
                x = data[:, 0]
                y = data[:, 2]
                params = self.fit_curve(x, y)
                temp_result['params'] = params
                temp_result['mt'] = mt_val
                temp_result['id'] = os.path.join(f, j)
                temp_result.update(other_input)
                self.dict_list.append(temp_result)
                
    def process_folder_name(self, folder_name):
        result_dict = {}
        fn_result = folder_name.split('_')
        cs_val = float(fn_result[0].replace('cs', ''))
        E_val = [float(i) for i in fn_result[1].replace('E', '').split('-')]
        L_val = [float(i) for i in fn_result[2].replace('L', '').replace('(base)', '').split('-')]
        result_dict['cs'] = cs_val
        result_dict['E'] = E_val
        result_dict['L'] = L_val
        return result_dict
        
    def process_file_name(self, file_name):
        mt = file_name.split('_')[1]
        mt = int(mt.replace('mT.txt', ''))
        return mt
        
    def read_txt(self, file_name):
        data = []
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                line_list = list(f.readlines())
        except UnicodeDecodeError:
            with open(file_name, 'r', encoding='latin1') as f:
                line_list = list(f.readlines())

        for i, line in enumerate(line_list):
            if i < 8:
                continue
            line_data = re.split(r'\s+', line)
            line_data = [float(j) for j in line_data[:7]]
            data.append(line_data)
        return data
    
    def fit_curve(self, x, y):
        params, _ = curve_fit(quadratic_curve, x, y)
        return params