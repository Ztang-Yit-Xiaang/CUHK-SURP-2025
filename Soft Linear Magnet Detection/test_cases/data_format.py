
import numpy as np

def load_npy():
    file_name = "./dataset/sensor_data.npy"
    # file_name = "./dataset/real_30_30.npy"
    data = np.load(file_name)
    print(data)
    print(data.shape)

def load_txt():
    file_name = "./dataset/save_sequence_file1.txt"
    data = np.loadtxt(file_name, dtype=str, delimiter=",")
    sensor_data = data[:, 1:].astype(float)
    print(sensor_data)
    print(sensor_data.shape)

if __name__ == "__main__":
    load_txt()
