
import numpy as np
import sys, argparse
sys.path.append("./")
from data_loader import MagneticDatasetSeq
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_data_modulus(cfg):
    '''
    模拟志愿者吞下胶囊，传感器阵列数值的变化情况
    '''
    # 1-载入传感器采集的数据
    train_dataset = MagneticDatasetSeq(dir_path=cfg.dataset_dir, file_name=cfg.key_word, data_len=225)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    data_all = []
    for data_sensor in train_loader:
        B_m = data_sensor.numpy().reshape(-1, 3)*1e-6   # uT -> T (real)

        B_m = np.abs(B_m)
        data_all = np.append(data_all, np.sum(B_m)*1e6)

    plt.plot(data_all)
    plt.show()

def remove_geo_bias():
    '''
    从原始数据去掉传感器的bias和环境的地磁场
    '''
    data_file_path = "./dataset/volunteer3/data_sequ_120919.txt"
    geo_file_path = "./dataset/volunteer3/data_once_120911.txt"
    nogeo_file_path = "./dataset/volunteer3/data_sequ_120919_nogeo.txt"

    raw_data = np.loadtxt(data_file_path, dtype=str, delimiter=",")
    geo_data = np.loadtxt(geo_file_path, delimiter=",")

    # 分离出时间戳
    timestamp = raw_data[:, 0]
    sensor_data = raw_data[:, 1:].astype(float)

    # 地磁取均值
    geo_mean = np.mean(geo_data, axis=0)

    nogeo_data = (sensor_data - geo_mean).astype(str)
    nogeo_data = np.insert(nogeo_data, 0, timestamp, axis=1)
    np.savetxt(nogeo_file_path, nogeo_data, fmt="%s", delimiter=",")

def args_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/test_rotation')
    parser.add_argument('--key_word', type=str, default="data_sequ_101923.txt")
    parser.add_argument('--save_dir', type=str, default="./results/")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = args_cfg()
    # show_data_modulus(cfg)
    remove_geo_bias()
