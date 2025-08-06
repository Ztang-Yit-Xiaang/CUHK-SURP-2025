'''
基于环形传感器阵列的姿态估计算法
- 铰链关节的旋转角度已知
'''
from scipy.optimize import least_squares
import numpy as np
import os
import argparse
import sys
sys.path.append(".")
from data_loader import MagneticDataset1D
from torch.utils.data import DataLoader
from timeit import default_timer as timer

def euler_to_mnp(alpha, beta, degrees=False):
    if degrees:
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
    m = np.sin(beta)
    n = -np.sin(alpha)*np.cos(beta)
    p = np.cos(alpha)*np.cos(beta)
    return (m, n, p)

def mnp_to_euler(m, n, p, degrees=False):
    alpha = np.arctan2(-n, p)
    beta = np.arcsin(m)
    if degrees:
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
    return (alpha, beta)

def get_mnp_error(mnp_true, mnp_pred, degrees=False):
    ori_dist = np.abs((mnp_true * mnp_pred).sum(axis=1)).reshape(-1, 1)
    ori_dist = np.minimum(ori_dist, 1)
    ori_err = np.arccos(ori_dist)
    if degrees:
        ori_err = np.rad2deg(ori_err)

    return ori_err

def get_alpha_error(alpha_true, alpha_pred):
    alpha_error1 = np.abs(alpha_true - alpha_pred)
    alpha_error2 = np.abs(alpha_true + alpha_pred)
    # alpha_error = np.where(alpha_error1<alpha_error2, alpha_error1, alpha_error2)
    alpha_error = np.minimum(alpha_error1, alpha_error2)
    return alpha_error

def get_beta_error(beta_true, beta_pred):
    beta_error = np.abs(beta_true - beta_pred)
    return beta_error

sensor_positions = np.array([
	[0.08, 0.08, 0.0005], [0.04, 0.08, 0.0005], [0.0, 0.08, 0.0005], [-0.04, 0.08, 0.0005], [-0.08, 0.08, 0.0005],
	[0.08, 0.04, 0.0005], [0.04, 0.04, 0.0005], [0.0, 0.04, 0.0005], [-0.04, 0.04, 0.0005], [-0.08, 0.04, 0.0005],
    [0.08, 0.00, 0.0005], [0.04, 0.00, 0.0005], [0.0, 0.00, 0.0005], [-0.04, 0.00, 0.0005], [-0.08, 0.00, 0.0005],
    [0.08,-0.04, 0.0005], [0.04,-0.04, 0.0005], [0.0,-0.04, 0.0005], [-0.04,-0.04, 0.0005], [-0.08,-0.04, 0.0005],
    [0.08,-0.08, 0.0005], [0.04,-0.08, 0.0005], [0.0,-0.08, 0.0005], [-0.04,-0.08, 0.0005], [-0.08,-0.08, 0.0005]
])

def fun_dipole_model(init_mag_prop, sensor_pos, B_m):
    # magnet_pos, magnet_ori, Bt = mag_prop
    a, b, c = init_mag_prop[0], init_mag_prop[1], init_mag_prop[2]
    m, n, p = init_mag_prop[3], init_mag_prop[4], init_mag_prop[5]
    Bt = init_mag_prop[6]
    Gx, Gy, Gz = init_mag_prop[7], init_mag_prop[8], init_mag_prop[9]

    B_pred = np.zeros(sensor_pos.shape)    # 25*3
    for i in range(sensor_pos.shape[0]):
        x = sensor_pos[i][0]
        y = sensor_pos[i][1]
        z = sensor_pos[i][2]
        
        R = np.sqrt( (x-a)*(x-a) + (y-b)*(y-b) + (z-c)*(z-c) )
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
        
        B_pred[i][0] = Bx_calc + Gx
        B_pred[i][1] = By_calc + Gy
        B_pred[i][2] = Bz_calc + Gz
    B_pred = B_pred.ravel()     # 25*3 -> 1 dimension
    B_m = B_m.ravel()
    error_arr = B_pred - B_m

    # 位姿约束
    error_ori = m**2 + n**2 + p**2 - 1
    error_arr = np.append(error_arr, error_ori)
    return error_arr

def pose_estimate(cfg):
    global sensor_positions
    
    # 载入传感器采集的数据
    train_dataset = MagneticDataset1D(dir_path=cfg.dataset_dir, key_word=cfg.key_word)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 载入上一轮传感器较正后的方向
    bt_file_path = os.path.join(cfg.save_dir, cfg.bt_save_name)
    ori_file_path = os.path.join(cfg.save_dir, cfg.ori_save_name)
    pos_file_path = os.path.join(cfg.save_dir, cfg.pos_save_name)
    Bt = np.median(np.load(bt_file_path))
    if os.path.exists(ori_file_path) and os.path.exists(pos_file_path):
        M = np.load(ori_file_path)
        sensor_positions = np.load(pos_file_path)
    else:
        print("[WARN] %s or %s do not exist." % (ori_file_path, pos_file_path))

    # 计算位置前的准备
    gt_mag_prop_arr = np.zeros((0, 10))
    est_mag_prop_arr = np.zeros((0, 10))
    for data_sensor, data_prop in train_loader:
        data_sensor = data_sensor[0, :].numpy()
        data_prop = data_prop[0, :].numpy()
        B_m = data_sensor.reshape(-1, 3) # 25*3

        # 磁铁真实位姿
        gt_mag_prop = data_prop

        # 磁铁初始值
        offset = np.array([0.03, -0.03, 0.03, 0.3, -0.3, 0.3, 0, 5e-6, 5e-6, 5e-6])
        init_mag_prop = gt_mag_prop + offset
        
        # 将传感器测量值从传感器坐标系变换到全局坐标系 (B_g = R_gs * B_s)
        for i in range(M.shape[0]):
            B_m[i] = np.dot(M[i], B_m[i].T).T   # 先转置成列与矩阵相乘, 结果再转置为行

        # 通过最小二乘法计算位姿
        tic = timer()
        res_lsq = least_squares(fun_dipole_model, init_mag_prop, method="lm", args=(sensor_positions, B_m))
        print("cost time: %f (s)" % (timer() - tic))

        # 结果处理
        if not res_lsq.success:
            print("success: ", res_lsq.success)
            continue
        est_mag_prop = res_lsq.x
        print("mag_prop: ", est_mag_prop)

        # 添加到列表中
        gt_mag_prop_arr = np.append(gt_mag_prop_arr, gt_mag_prop.reshape(-1, 10), axis=0)
        est_mag_prop_arr = np.append(est_mag_prop_arr, est_mag_prop.reshape(-1, 10), axis=0)
    
    # 精度计算(总的位置误差)
    pos_true = gt_mag_prop_arr[:, 0:3]*1e3
    pos_pred = est_mag_prop_arr[:, 0:3]*1e3     # m -> mm
    pos_error = np.linalg.norm(pos_true-pos_pred, axis=1)
    
    # 精度计算(总的方向误差)
    mnp_true = gt_mag_prop_arr[:, 3:6]
    mnp_pred = est_mag_prop_arr[:, 3:6]
    # ones = np.sqrt(mnp_pred[:, 0]**2 + mnp_pred[:, 1]**2 + mnp_pred[:, 2]**2)
    # ones = ones.reshape(-1, 1)
    # mnp_pred = mnp_pred/ones
    ori_error = get_mnp_error(mnp_true, mnp_pred, degrees=True)

    print("pos_error: mean=%.3f, std=%.3f, min=%.3f, max=%.3f" %
                    (np.mean(pos_error), np.std(pos_error), np.min(pos_error), np.max(pos_error) ))
    print("ori_error: mean=%.3f, std=%.3f, min=%.3f, max=%.3f" %
                    (np.mean(ori_error), np.std(ori_error), np.min(ori_error), np.max(ori_error)))
    return est_mag_prop_arr


def args_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--key_word', type=str, default="sensor")
    parser.add_argument('--save_dir', type=str, default="./results/")
    parser.add_argument('--bt_save_name', type=str, default="bt.npy")
    parser.add_argument('--ori_save_name', type=str, default="sensor_ori.npy")
    parser.add_argument('--pos_save_name', type=str, default="sensor_pos.npy")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = args_cfg()

    est_mag_pose = pose_estimate(cfg)
    # print("x: \r\n", est_mag_pose)

    # 测试euler与mnp之间的转换
    # alpha, beta = -130, -90
    # m, n, p = euler_to_mnp(alpha, beta, degrees=True)
    # a, b = mnp_to_euler(m, n, p, degrees=True)
    # print(a, b)