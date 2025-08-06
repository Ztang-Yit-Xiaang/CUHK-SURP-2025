'''
模拟生成环形传感器阵列的数据
- 根据磁偶极子模型, 计算全局坐标系下传感器测量值(模拟)
- 根据传感器姿态(传感器坐标系相对于全局坐标系), 变换得到传感器坐标系下的传感器测理值
'''
import numpy as np
import os, sys, math
import argparse

sys.path.append(".")

def Bt_calc(a=10*1e-3, r=5*1e-3, M=890000):
    a = a                  # length of the cylinder magnet (m)
    r = r                  # radius (m)
    M = M                  # Magnetization
    Vm = math.pi * r**2 * a     # volume of the cylinder magnet

    mu_0 = 4*math.pi*10**-7     # 
    mu_r = 1.0997785406    # relative permeability
    Bt = mu_0 * mu_r * M * Vm / (4*math.pi)

    return Bt


def dipole_model(sensor_pos, magnet_pos, magnet_ori, Bt):
    '''
    磁偶极子模型
    输入: 
    -sensor_pos: 传感器阵列每个传感器的位置
    -magnet_pos: 磁铁的位置
    -magnet_ori: 磁铁的姿态
    -Bt: 磁性强度常量
    输出: 磁场强度(磁通量密度)
    '''
    a, b, c = magnet_pos[0], magnet_pos[1], magnet_pos[2]
    m, n, p = magnet_ori[0], magnet_ori[1], magnet_ori[2]

    B_pred = np.zeros(sensor_pos.shape)
    for i in range(sensor_pos.shape[0]):
        x = sensor_pos[i][0]
        y = sensor_pos[i][1]
        z = sensor_pos[i][2]
        
        R = math.sqrt( (x-a)*(x-a) + (y-b)*(y-b) + (z-c)*(z-c) )
        R3 = R**3
        R5 = R**5

        Bx1 = 3*(m*(x-a)+n*(y-b)+p*(z-c))*(x-a)/R5
        Bx2 = m/R3
        Bx_calc = Bt*(Bx1-Bx2)

        By1 = 3*(m*(x-a)+n*(y-b)+p*(z-c))*(y-b)/R5
        By2 = n/R3
        By_calc = Bt*(By1-By2)

        Bz1 = 3*(m*(x-a)+n*(y-b)+p*(z-c))*(z-c)/R5
        Bz2 = p/R3
        Bz_calc = Bt*(Bz1-Bz2)
        
        B_pred[i][0] = Bx_calc
        B_pred[i][1] = By_calc
        B_pred[i][2] = Bz_calc
    return B_pred

def args_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--key_word', type=str, default="sensor")
    parser.add_argument('--save_dir', type=str, default="./results/")
    parser.add_argument('--bt_save_name', type=str, default="bt.npy")
    parser.add_argument('--ori_save_name', type=str, default="sensor_ori.npy")
    parser.add_argument('--pos_save_name', type=str, default="sensor_pos.npy")
    parser.add_argument('--data_save_name', type=str, default="sensor_data.npy")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = args_cfg()

    Bt = Bt_calc()
    bt_file_path = os.path.join(cfg.save_dir, cfg.bt_save_name)
    np.save(bt_file_path, Bt)

    # 磁铁位姿
    magnet_pos = np.array([0, 0, 70]) * 1e-3      # unit: m
    magnet_ori = np.array([0, 0, 1])
    Geo = np.array([[11, 0, -8]])*1e-6

    # 0-载入环形传感器阵列的位置和姿态
    sensor_pos = np.load(os.path.join(cfg.save_dir, cfg.pos_save_name))
    sensor_ori = np.load(os.path.join(cfg.save_dir, cfg.ori_save_name))

    # 1-全局坐标系下的传感器测量值
    B_g = dipole_model(sensor_pos, magnet_pos, magnet_ori, Bt)

    # 1.1-叠加地磁分量
    Geo_arr = np.repeat(Geo, B_g.shape[0], axis=0)
    # B_g = B_g + Geo_arr
    B_g = Geo_arr
    print("B_g = \n", B_g)

    # 2-变换得到传感器坐标系下的传感器测量值
    B_s = np.zeros((0, 3))
    for i in range(len(sensor_ori)):
        B_gi = B_g[i].reshape(3, 1)             # 取出某一个传感器数据
        R_gs = sensor_ori[i]                    # 传感器的姿态
        B_temp = np.transpose(R_gs).dot(B_gi).T # 变换到传感器坐标系下
        B_s = np.append(B_s, B_temp, axis=0)

    # 3-变更传感器数据的格式: 一行对应一帧
    print("Before: \n", B_s)
    B_s = B_s.reshape(1, -1)
    data = np.append(B_s, np.concatenate((magnet_pos, magnet_ori)))
    data = np.append(data, Bt)
    data = np.append(data, Geo)
    print("After: \n", data*1e6)

    data_file_path = os.path.join(cfg.dataset_dir, cfg.data_save_name)
    np.save(data_file_path, data)
