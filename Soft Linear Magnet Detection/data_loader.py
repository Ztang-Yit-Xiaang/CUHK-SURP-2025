import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import time
from glob import glob

class MagneticDataset1D(Dataset):
    '''
    数据中带有真值
    '''
    def __init__(self, dir_path, key_word) -> None:
        super().__init__()
        # 数据加载及预处理
        file_list = glob('{:s}/{:s}*'.format(dir_path, key_word))
        file_list.sort()

        data_all = np.zeros((0, 255+7))     # 195 + 3 pos + 3 ori + 1 Bt
        for file_name in file_list:
            # 1-compatible with .npy and txt files
            if file_name.endswith(".npy"):
                data = np.load(file_name)
            else:
                data = np.loadtxt(file_name, delimiter=',')

            # 只一行数据的文件, 进行维度扩展
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            data_all = np.append(data_all, data, axis=0)
        print("data_all shape: ", data_all.shape)
        data_sensor = data_all[:, 0:255]

        # 相关变量初始化
        self.len = data_all.shape[0]
        # self.len = 400
        self.data_sensor = torch.from_numpy(data_sensor).float()
        self.data_prop = torch.from_numpy(data_all[:, 255:]).float()
        print("data_sensor shape", self.data_sensor.shape)
        print("*"*40)
        # input("Press Enter to continue...")
    
    def __getitem__(self, index):
        return self.data_sensor[index], self.data_prop[index]

    def __len__(self):
        return self.len

class MagneticDatasetSeq(Dataset):
    '''
    数据中没有带真值
    '''
    def __init__(self, dir_path, file_name, data_len, idx_offset=0) -> None:
        super().__init__()
        # 数据加载及预处理
        file_path = os.path.join(dir_path, file_name)

        if file_name.endswith(".txt"):
            data = np.loadtxt(file_path, dtype=str, delimiter=',')
            self.len = data.shape[0] - idx_offset   # 要把偏移扣除
            if(data.shape[1] == data_len+1):
                self.timestamp = data[idx_offset:, 0]
                self.data_sensor = data[idx_offset:, 1:].astype(float)    # 去除第一个时间戳
            else:
                self.timestamp = np.repeat(np.array(["12:00:01.0"]), self.len)
                self.data_sensor = data.astype(float)
        else:
            print("It does not txt file, please check the file name")
    
    def __getitem__(self, index):
        return (self.timestamp[index], self.data_sensor[index])

    def __len__(self):
        return self.len

if __name__ == "__main__":

    train_dataset = MagneticDataset1D(dir_path="./dataset", key_word="sensor")
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    for data_sensor, data_prop in train_loader:
        print("data_sensor shape: ", data_sensor.shape)
        print("data_prop shape: ", data_prop.shape)
