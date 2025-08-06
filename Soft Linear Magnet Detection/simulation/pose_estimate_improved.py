'''
基于环形传感器阵列的姿态估计算法
- 铰链关节的旋转角度未知
'''
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import argparse
from .data_loader import MagneticDataset1D
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from .ring_array_model import get_linear_flexible_array_pose, get_spatial_flexible_array_pose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def ring_array_pos_ext(h1_pos, h1_beta, h12_len, s_height):
    '''
    沿x轴正方向扩展
    问题描述: 在全局坐标系下, 从位置hinge1_pos开始, 沿方向beta平移len长度后, 到达
             位置点hinge2_pos, 求pos_2的坐标?
    输入: 
    -h1_pos:   前一个铰链的位置
    -h1_beta:  前一个铰链的旋转角度, 即绕Z轴旋转的角度(相对于全局坐标系, 而不是上一个模块的局部坐标系)
    -h12_len:  两个铰链之间的长度
    -s_height: 传感器离铰链平面的高度
    
    输出: 下一个铰链的位置和姿态
    '''
    p_to_hinge = np.array([[h12_len, 0, 0]]).T              # 指向下一个铰链的向量
    p_to_sensor = np.array([[h12_len/2, 0, s_height]]).T    # 指向传感器的向量
    theta = [0, 0, h1_beta]            # 沿Z轴旋转
    R_12 = R.from_euler('XYZ', theta, degrees=True)

    # 计算铰链2的位置
    h2_pos = h1_pos + R_12.as_matrix().dot(p_to_hinge)

    # 计算传感器的位置
    sensor_pos = h1_pos + R_12.as_matrix().dot(p_to_sensor)

    # 沿Z方向扩展, 形成条状传感器阵列
    sensor_pos_arr = np.zeros((3, 0))
    offset = np.array([[0, 0.00, 0.05]]).T     # 条状传感器阵列中，传感器间距为50 mm
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*2, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*0, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*2, axis=1)
    
    # 姿态扩展, 采用expand_dims在0维度上增加一个维度, 再在0维度上重复5次
    sensor_ori_arr = np.expand_dims(R_12.as_matrix(), 0).repeat(5, axis=0)

    return sensor_pos_arr.T, sensor_ori_arr, h2_pos

def ring_array_neg_ext(h1_pos, h1_beta, h12_len, s_height):
    '''
    沿x轴负方向扩展
    问题描述: 在全局坐标系下, 从位置hinge1_pos开始, 沿方向beta平移len长度后, 到达
             位置点hinge2_pos, 求pos_2的坐标?
    输入: 
    -h1_pos:   前一个铰链的位置
    -h1_beta:  前一个铰链的旋转角度, 即绕Z轴旋转的角度(相对于全局坐标系, 而不是上一个模块的局部坐标系)
    -h12_len:  两个铰链之间的长度
    -s_height: 传感器离铰链平面的高度
    
    输出: 下一个铰链的位置和姿态
    '''
    p_to_hinge = np.array([[-h12_len, 0, 0]]).T              # 指向下一个铰链的向量(与正方向的不同点)
    p_to_sensor = np.array([[-h12_len/2, 0, s_height]]).T    # 指向传感器的向量(与正方向的不同点)
    theta = [0, 0, h1_beta]            # 沿Z轴旋转
    R_12 = R.from_euler('XYZ', theta, degrees=True)

    # 计算铰链2的位置
    h2_pos = h1_pos + R_12.as_matrix().dot(p_to_hinge)

    # 计算传感器的位置
    sensor_pos = h1_pos + R_12.as_matrix().dot(p_to_sensor)

    # 沿Z方向扩展, 形成条状传感器阵列
    sensor_pos_arr = np.zeros((3, 0))
    offset = np.array([[0, 0, 0.05]]).T     # 条状传感器阵列中，传感器间距为50 mm
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*2, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*0, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*2, axis=1)
    
    # 姿态扩展, 采用expand_dims在0维度上增加一个维度, 再在0维度上重复5次
    sensor_ori_arr = np.expand_dims(R_12.as_matrix(), 0).repeat(5, axis=0)

    return sensor_pos_arr.T, sensor_ori_arr, h2_pos

def get_array_pose(beta1, beta2):
    main_unit_width = 0.05      # 每个单元的宽度为50mm (两个铰链之间的长度)
    slave_unit_width = 0.07
    sensor_height = -0.01  # 传感器离铰链平面的高度 (负号表示沿Z轴负方向)
    sensor_pos_all = np.zeros((0, 3))
    sensor_ori_all = np.zeros((0, 3, 3))

    # 主控板(基板)
    sensor_pos_base, sensor_ori_base, hinge_pos_base = \
        ring_array_pos_ext(np.array([[0, 0, 0]]).T, 0, 0, sensor_height)
    sensor_pos_all = np.append(sensor_pos_all, sensor_pos_base, axis=0)
    sensor_ori_all = np.append(sensor_ori_all, sensor_ori_base, axis=0)
    # print("sensor_pos_base: \n", sensor_pos_base)
    # print("sensor_ori_base: \n", sensor_ori_base)

    # 沿X负方向扩展
    _, _, hinge_pi_pos = ring_array_neg_ext(hinge_pos_base, 0, main_unit_width/2, sensor_height)       # 注意/2
    for beta in np.array([0, 0]):
        sensor_pj_pos, sensor_pj_ori, hinge_pj_pos = \
            ring_array_neg_ext(hinge_pi_pos, beta, main_unit_width, sensor_height)
        sensor_pos_all = np.append(sensor_pos_all, sensor_pj_pos, axis=0)
        sensor_ori_all = np.append(sensor_ori_all, sensor_pj_ori, axis=0)
        hinge_pi_pos = hinge_pj_pos

    for beta in beta1:
        sensor_pj_pos, sensor_pj_ori, hinge_pj_pos = \
            ring_array_neg_ext(hinge_pi_pos, beta, slave_unit_width, sensor_height)
        sensor_pos_all = np.append(sensor_pos_all, sensor_pj_pos, axis=0)
        sensor_ori_all = np.append(sensor_ori_all, sensor_pj_ori, axis=0)
        hinge_pi_pos = hinge_pj_pos

    # 沿X正方向扩展
    _, _, hinge_pi_pos = ring_array_pos_ext(hinge_pos_base, 0, main_unit_width/2, sensor_height)     # 注意/2
    for beta in np.array([0, 0]):
        sensor_pj_pos, sensor_pj_ori, hinge_pj_pos = \
            ring_array_pos_ext(hinge_pi_pos, beta, main_unit_width, sensor_height)
        sensor_pos_all = np.append(sensor_pos_all, sensor_pj_pos, axis=0)
        sensor_ori_all = np.append(sensor_ori_all, sensor_pj_ori, axis=0)
        hinge_pi_pos = hinge_pj_pos

    for beta in beta2:
        sensor_pj_pos, sensor_pj_ori, hinge_pj_pos = \
            ring_array_pos_ext(hinge_pi_pos, beta, slave_unit_width, sensor_height)
        sensor_pos_all = np.append(sensor_pos_all, sensor_pj_pos, axis=0)
        sensor_ori_all = np.append(sensor_ori_all, sensor_pj_ori, axis=0)
        hinge_pi_pos = hinge_pj_pos

    # print("sensor_pos_all: \n", sensor_pos_all)
    # print("sensor_ori_all: \n", sensor_ori_all)
    return sensor_pos_all, sensor_ori_all
def calc_dipole_field(sensor_pos, dipole_pos, dipole_ori, Bt):
    """
    單一 dipole 的磁場計算（根據磁偶極子模型）
    sensor_pos: 感測器位置 (3,)
    dipole_pos: dipole 位置 (3,)
    dipole_ori: dipole 方向單位向量 (3,)
    Bt: 磁強度
    return: 磁場 (3,)
    """
    a, b, c = dipole_pos
    m, n, p = dipole_ori
    x, y, z = sensor_pos

    R_vec = np.array([x - a, y - b, z - c])
    R = np.linalg.norm(R_vec)
    if R < 1e-6:
        return np.zeros(3)  # 避免除以0

    R3 = R**3
    R5 = R**5

    dot = m*(x-a) + n*(y-b) + p*(z-c)
    Bx = Bt * (3*dot*(x-a)/R5 - m/R3)
    By = Bt * (3*dot*(y-b)/R5 - n/R3)
    Bz = Bt * (3*dot*(z-c)/R5 - p/R3)

    return np.array([Bx, By, Bz])


def fun_dipole_model(init_mag_prop, sensor_data, frame_idx, cfg = None, return_full=False):
    '''
    注意点: 
    1. 不要在该函数内更改传递参数sensor_data的值, 该函数会在迭代中循环调用, 
       假如这次更改了sensor_data值, 下次迭代会是变更过的值
    2. 
    '''
    # magnet_pos, magnet_ori, Bt = mag_prop
    a, b, c = init_mag_prop[0:3]#磁鐵位置
    m, n, p = init_mag_prop[3:6]#磁鐵方向
    Bt = init_mag_prop[6]
    Gx, Gy, Gz = init_mag_prop[7:10]

    # 预测环形传感器阵列的位置和方向
    beta1, beta2 = init_mag_prop[10:15], init_mag_prop[15:20]
    sensor_pos, sensor_ori = get_array_pose(beta1, beta2)
    # 用 beta1, beta2 計算 dipole 分布
    start_pos = np.array([a, b, c]).reshape(3, 1)  # 磁铁位置
    start_dir = np.array([m, n, p]).reshape(3, 1)  # 磁铁方向
    start_dir = start_dir / np.linalg.norm(start_dir)
    if cfg is not None:
        filename = cfg.key_word
        if not filename.endswith(".npy"):
            filename += ".npy"
        data_path = os.path.join(cfg.dataset_dir, filename)
    else:
        data_path = "dataset/captured_20250731_161948.npy"

    bend_angles = estimate_bending_from_captured_data(data_path=data_path,
                                                       num_segments=4,
                                                       frame_idx=frame_idx)

    # 获取线性柔性阵列的姿态
    # 注意: 这里的 start_pos 和 start_dir 是磁铁的初始位置
    #       而不是传感器阵列的初始位置get_spatial_flexible_array_pose
    #       传感器阵列的初始位置是通过 beta1, beta2 计算得到的
    dipole_pos, dipole_ori = get_linear_flexible_array_pose(start_pos, start_dir, bend_angles=bend_angles, num_segments=len(bend_angles))
    # phi_angles = np.linspace(10, -10, len(bend_angles))  # 或全 0 表示平面內彎曲
    # 获取柔性阵列的空间位置和方向
    # dipole_pos, dipole_ori = get_spatial_flexible_array_pose(
    # start_pos, start_dir,
    # bend_angles=bend_angles,
    # phi_angles=phi_angles,
    # segment_len=0.01
    #   )
    # 将传感器测量值从传感器坐标系变换到全局坐标系 (B_g = R_gs * B_s)
    B_m = np.zeros_like(sensor_data)
    for i in range(sensor_ori.shape[0]):
        R_gs = sensor_ori[i]  # 传感器 i 的方向矩阵
        # 将传感器测量值从传感器坐标系变换到全局坐标系
        B_m[i,:] = np.dot(R_gs, sensor_data[i])  # 注意转置，确保维度匹配


    B_pred = np.zeros_like(sensor_data)    # 25*3
    for i in range(sensor_pos.shape[0]):
        B_total = np.zeros(3)
        for j in range(dipole_ori.shape[0]):
            B_seg = calc_dipole_field(sensor_pos[i],
                                        dipole_pos[j],
                                        dipole_ori[j],
                                        Bt/dipole_pos.shape[0])
            B_total += B_seg
        B_total += np.array([Gx, Gy, Gz])  # 添加地磁偏移
        B_pred[i] = B_total
    if return_full:
        print("dipole_pos[0] =", dipole_pos[0])
        return B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori
    # 整理成向量
    B_pred = B_pred.ravel()     # 25*3 -> 1 dimension
    B_m = B_m.ravel()

    
    # 位姿约束
    error_arr = B_pred - B_m
    error_ori = m**2 + n**2 + p**2 - 1
    error_arr = np.append(error_arr, error_ori)
    # ✅ 額外加上鼓勵 Y 展開的 penalty 項
    # y_std = np.std(dipole_pos[:, 1])  # Y 軸標準差
    # penalty = -0.02 * y_std           # 負值 → 誘導最小化時去放大 y_std
    # spread_penalty = -0.01 * np.linalg.norm(np.std(dipole_pos, axis=0))
    # error_arr = np.append(error_arr, spread_penalty)
    print("Total loss (L2):", np.linalg.norm(error_arr))

    return error_arr

def estimate_bending_from_captured_data(data_path, num_segments=12, frame_idx=0):
    """
    根據 captured_data.npy 中的磁場向量估算撓曲角度分佈
    - data_path: .npy 路徑
    - num_segments: 要估算的分段數
    - frame_idx: 使用哪一個時間點的資料進行估算（預設第0幀）
    
    回傳：估算的 bend_angles 陣列 (num_segments,)
    """
    data = np.load(data_path)  # (T, 85, 3)
    if data.ndim == 2 and data.shape[1] == 255:
        data = data.reshape(-1, 85, 3)
    if frame_idx >= data.shape[0]:
        raise IndexError("❌ 指定的 frame index 超出資料長度")

    B_frame = data[frame_idx:frame_idx+1]  # shape: (1, 85, 3)

    sensors = B_frame.shape[1]
    seg_len = sensors // num_segments
    bend_angles = []

    for i in range(num_segments):
        start = i * seg_len
        end = (i+1) * seg_len if i < num_segments - 1 else sensors
        B_seg = B_frame[:, start:end, :]
        vec = np.mean(B_seg, axis=(0, 1))
        vec /= np.linalg.norm(vec) + 1e-8
        angle = np.arccos(vec[2]) #  # 计算与Z轴的夹角
        bend_angles.append(np.rad2deg(angle))
        #print("segment {} mean_vec = {}, angle = {}".format(i, vec, angle))
    # 如果有剩余的传感器数据，计算最后一段
    if sensors % num_segments != 0:
        start = num_segments * seg_len
        B_seg = B_frame[:, start:, :]
        vec = np.mean(B_seg, axis=(0, 1))
        vec /= np.linalg.norm(vec) + 1e-8
        angle = np.arccos(vec[2])
        bend_angles.append(np.rad2deg(angle))
        #print("segment {} mean_vec = {}, angle = {}".format(num_segments, vec, angle))
    #print(f"Frame {frame_idx}, bend_angles = {bend_angles}")

    return np.array(bend_angles)

def pose_estimate(cfg):
    
    # 载入传感器采集的数据
    train_dataset = MagneticDataset1D(dir_path=cfg.dataset_dir, key_word=cfg.key_word)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 计算位置前的准备
    gt_mag_prop_arr = np.zeros((0, 10))
    est_mag_prop_arr = np.zeros((0, 10))
    prev_mag_prop = None  # ⭐ 用來儲存前一幀估值

    for data_sensor, data_prop in train_loader:
        data_sensor = data_sensor[0, :].numpy()
        data_prop = data_prop[0, :].numpy()
        B_m = data_sensor.reshape(-1, 3) # 25*3

        # 磁铁真实位姿
        gt_mag_prop = data_prop

        # 初始值估计
        # beta1 = np.array([45.0,  90.0,  90.0,  120.0, 150, 180]) + np.array([5, 5, -10, 10, 10, -10])
        # beta2 = np.array([-45.0,  -90.0, -90.0, -120.0, -150, -180]) + np.array([5, 5, -10, 10, -10, 10])
        beta1 = np.array([65,  90,  120,  160, 180]) + np.array([0, 2, -3, 1, -2])
        beta2 = np.array([-50,  -90,  -125,  -150, -180]) + np.array([1, -1, 2, -3, 0])
        offset = np.array([0.03, 0.03, -0.03, 0.3, 0.3, -0.3, 0, 5e-6, 5e-6, 5e-6])
        if prev_mag_prop is None:
            init_mag_prop = gt_mag_prop + offset
            init_mag_prop = np.append(init_mag_prop, np.concatenate((beta1, beta2)))
        else:
            init_mag_prop = prev_mag_prop.copy()

        print("init_mag_prop shape:", init_mag_prop.shape)
        print("init_mag_prop: ", init_mag_prop)

        # 通过最小二乘法计算位姿，并計算用時
        tic = timer()
        res_lsq = least_squares(fun_dipole_model, init_mag_prop, method="lm", args=(B_m,))
        print("cost time: %f (s)" % (timer() - tic))

        # 优化结果处理
        if not res_lsq.success:
            print("success: ", res_lsq.success)
            exit()
        est_mag_prop = res_lsq.x
        prev_mag_prop = est_mag_prop.copy()  # 保存当前估计结果作为下一帧的初始值
        print("mag_prop: ", est_mag_prop)
        est_mag_prop = est_mag_prop[0:10]       # 取部分结果,磁鐵的位姿與場强

        # 添加到列表中
        gt_mag_prop_arr = np.append(gt_mag_prop_arr, gt_mag_prop.reshape(-1, 10), axis=0)
        est_mag_prop_arr = np.append(est_mag_prop_arr, est_mag_prop.reshape(-1, 10), axis=0)
    
    # 精度计算(总的位置误差)
    pos_true = gt_mag_prop_arr[:, 0:3]*1e3
    pos_pred = est_mag_prop_arr[:, 0:3]*1e3     # m -> mm
    pos_error = np.linalg.norm(pos_true-pos_pred, axis=1)
    
    if pos_error.size == 0:
        print("❌ 無估算結果：無可用樣本或資料集為空。請確認資料載入與過濾邏輯。")
        return None


    # 精度计算(总的方向误差)
    mnp_true = gt_mag_prop_arr[:, 3:6]
    mnp_pred = est_mag_prop_arr[:, 3:6]
    # ones = np.sqrt(mnp_pred[:, 0]**2 + mnp_pred[:, 1]**2 + mnp_pred[:, 2]**2)
    # ones = ones.reshape(-1, 1)
    # mnp_pred = mnp_pred/ones
    ori_error = get_mnp_error(mnp_true, mnp_pred, degrees=True)

    print("pos_error: mean=%.3f(mm), std=%.3f, min=%.3f, max=%.3f" %
                    (np.mean(pos_error), np.std(pos_error), np.min(pos_error), np.max(pos_error) ))
    print("ori_error: mean=%.3f(°), std=%.3f, min=%.3f, max=%.3f" %
                    (np.mean(ori_error), np.std(ori_error), np.min(ori_error), np.max(ori_error)))
    return est_mag_prop_arr

def pose_estimate_single(cfg, frame_idx=0):
    print(f"🔎 Frame {frame_idx}")
    """
    仅估计指定帧的磁铁参数。
    """
    dataset = MagneticDataset1D(cfg.dataset_dir, cfg.key_word)
    if frame_idx >= len(dataset):
        raise IndexError(f"frame_idx 超出数据长度：{frame_idx} >= {len(dataset)}")

    sensor_data, _ = dataset[frame_idx]  # shape: (255,)
    print(f"Frame {frame_idx}, mean field = {sensor_data.mean(axis=0)}")
    sensor_data = sensor_data.reshape(85, 3)
    sensor_data = sensor_data[:75, :]  # use only the valid ones
    print("B mean (raw) =", sensor_data.mean())
    print("B max =", np.max(sensor_data))
    print("B min =", np.min(sensor_data))

    # 先猜一个初始值
    init_mag_prop = np.array([
        0.02, -0.01, -0.04,     # 磁铁位置
        1.00, -0.30, 1.00,      # 磁铁方向
        0.0,                   # Bt
        5e-6, 5e-6, 5e-6       # 地磁偏移
    ])
    #init_mag_prop[3:6] = np.array([0.5, 0, 0.8]) / np.linalg.norm([0.5, 0, 0.8])

    # beta1 = np.array([50, 95, 80, 130, 160, 170])
    # beta2 = np.array([-40, -85, -100, -110, -160, -170])
    beta1 = np.array([65,  90,  120,  160, 180]) + np.array([0, 2, -3, 1, -2])
    beta2 = np.array([-50,  -90,  -125,  -150, -180]) + np.array([1, -1, 2, -3, 0])
    init_mag_prop = np.concatenate((init_mag_prop, beta1, beta2))

    res_lsq = least_squares(fun_dipole_model, init_mag_prop, args=(sensor_data, frame_idx, cfg), method="lm")
    est_mag_prop = res_lsq.x[:-10]
    B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori = fun_dipole_model(est_mag_prop, sensor_data, return_full=True, frame_idx=frame_idx, cfg=cfg)
    for i in range(sensor_pos.shape[0]):
        print(f"{i:2d}: Pos={sensor_pos[i]}, Ori_z={sensor_ori[i][:,2]}")

    #print("sensor_ori.shape =", sensor_ori.shape)

    return B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori  # 前十個个参数（估计结果）


def args_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--key_word', type=str, default="captured_20250731_161948.npy")
    parser.add_argument('--save_dir', type=str, default="./results/")
    parser.add_argument('--bt_save_name', type=str, default="bt.npy")
    parser.add_argument('--ori_save_name', type=str, default="sensor_ori.npy")
    parser.add_argument('--pos_save_name', type=str, default="sensor_pos.npy")
    return parser.parse_args()

if __name__ == "__main__":
    
    cfg = args_cfg()
    print("🔍 嘗試讀取資料夾: ", cfg.dataset_dir)

    est_mag_pose = pose_estimate(cfg)
    # print("x: \r\n", est_mag_pose)

    # 测试euler与mnp之间的转换
    # alpha, beta = -130, -90
    # m, n, p = euler_to_mnp(alpha, beta, degrees=True)
    # a, b = mnp_to_euler(m, n, p, degrees=True)
    # print(a, b)