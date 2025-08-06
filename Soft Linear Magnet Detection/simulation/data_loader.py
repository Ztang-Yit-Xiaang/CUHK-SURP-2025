import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import time
from glob import glob

class MagneticDataset2D(Dataset):
    def __init__(self, dir_path, key_word) -> None:
        super().__init__()
        # 数据加载及预处理
        file_list = glob('{:s}/{:s}*'.format(dir_path, key_word))
        file_list.sort()

        data_all = np.zeros((0, 85, 3))        # 3 pos + 3 euler angle + 1 bt
        for file_name in file_list:
            # 1-compatible with .npy and txt files
            if file_name.endswith(".npy"):
                data = np.load(file_name)
            else:
                data = np.loadtxt(file_name, delimiter=',')

            # 2-compatible with 81 and 82 cols
            if data.shape[1] == 81:
                print("[Warning] extend col dim of data from 81 to 82")
                data_ext = np.zeros((data.shape[0], 1))
                data = np.append(data, data_ext, axis=1)
            elif data.shape[1] == 75:   # for interference
                print("[Warning] extend col dim of data from 75 to 82")
                data_ext = np.zeros((data.shape[0], 7))
                data = np.append(data, data_ext, axis=1)
            data_all = np.append(data_all, data, axis=0)
        print("data_all shape: ", data_all.shape)
        data_sensor = data_all[:, 0:75]
        data_prop = data_all[:, 75:]

        data_sensor = data_sensor.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
        
        # 相关变量初始化
        self.len = data_all.shape[0]
        # self.len = 100
        self.data_sensor = torch.from_numpy(data_sensor).float()
        self.data_prop = torch.from_numpy(data_prop).float()     # pose+Bt -> property
        print("data_sensor shape", self.data_sensor.shape)
        print("*"*40)
        # time.sleep(2)
        # input("Press Enter to continue...")
    
    def __getitem__(self, index):
        return self.data_sensor[index], self.data_prop[index]

    def __len__(self):
        return self.len


class MagneticDataset1D(Dataset):
    def __init__(self, dir_path='./dataset', key_word='captured_data'):
        self.data = []
        self.prop = []

        for file in os.listdir(dir_path):
            if file.endswith(".npy") and key_word in file:
                data_path = os.path.join(dir_path, file)
                data = np.load(data_path)

                if data.ndim == 3 and data.shape[1:] == (85, 3):
                    self.data.append(data.reshape(data.shape[0], -1))  # (N, 85, 3) → (N, 255)
                    print(f"加載文件: {file}, 形狀: {data.shape}")
                elif data.ndim == 2 and data.shape[1] == 255:
                    self.data.append(data)
                    print(f"加載展平文件: {file}, 形狀: {data.shape}")
                else:
                    print(f"文件 {file} 形狀不符: {data.shape}")
                    continue

                

        

        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
        else:
            self.data = np.empty((0, 255))

        # 若沒有 ground truth prop，就填零佔位
        self.prop = np.zeros((self.data.shape[0], 10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sensor = self.data[idx]
        prop = self.prop[idx]
        return sensor.astype(np.float32), prop.astype(np.float32)


if __name__ == "__main__":

    train_dataset = MagneticDataset1D(dir_path="./dataset", key_word="sensor")
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    for data_sensor, data_prop in train_loader:
        print("data_sensor shape: ", data_sensor.shape)
        print("data_prop shape: ", data_prop.shape)
