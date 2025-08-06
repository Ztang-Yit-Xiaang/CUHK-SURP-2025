'''
基于环形传感器阵列的姿态估计算法
- 铰链关节的旋转角度未知
- 数据集中没有带真值
'''
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy as np
import os, sys, argparse
from data_loader import MagneticDatasetSeq
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

def ring_array_pos_ext(h1_pos, h1_beta, h12_len, s_height):
    '''
    沿x轴正方向扩展
    问题描述: 在全局坐标系下, 从位置hinge1_pos开始, 沿方向beta平移len长度后, 到达
             位置点hinge2_pos, 求pos_2的坐标?
    输入: 
    -h1_pos:   前一个铰链的位置
    -h1_beta:  前一个铰链的旋转角度, 即绕Y轴旋转的角度(相对于全局坐标系, 而不是上一个模块的局部坐标系)
    -h12_len:  两个铰链之间的长度
    -s_height: 传感器离铰链平面的高度
    
    输出: 下一个铰链的位置和姿态
    '''
    p_to_hinge = np.array([[h12_len, 0, 0]]).T              # 指向下一个铰链的向量
    p_to_sensor = np.array([[h12_len/2, 0, s_height]]).T    # 指向传感器的向量
    theta = [0, h1_beta, 0]            # 沿Y轴旋转
    R_12 = R.from_euler('XYZ', theta, degrees=True)

    # 计算铰链2的位置
    h2_pos = h1_pos + R_12.as_matrix().dot(p_to_hinge)

    # 计算传感器的位置
    sensor_pos = h1_pos + R_12.as_matrix().dot(p_to_sensor)

    # 沿Y方向扩展, 形成条状传感器阵列
    sensor_pos_arr = np.zeros((3, 0))
    offset = np.array([[0, 0.05, 0]]).T     # 条状传感器阵列中，传感器间距为50 mm
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
    -h1_beta:  前一个铰链的旋转角度, 即绕Y轴旋转的角度(相对于全局坐标系, 而不是上一个模块的局部坐标系)
    -h12_len:  两个铰链之间的长度
    -s_height: 传感器离铰链平面的高度
    
    输出: 下一个铰链的位置和姿态
    '''
    p_to_hinge = np.array([[-h12_len, 0, 0]]).T              # 指向下一个铰链的向量(与正方向的不同点)
    p_to_sensor = np.array([[-h12_len/2, 0, s_height]]).T    # 指向传感器的向量(与正方向的不同点)
    theta = [0, h1_beta, 0]            # 沿Y轴旋转
    R_12 = R.from_euler('XYZ', theta, degrees=True)

    # 计算铰链2的位置
    h2_pos = h1_pos + R_12.as_matrix().dot(p_to_hinge)

    # 计算传感器的位置
    sensor_pos = h1_pos + R_12.as_matrix().dot(p_to_sensor)

    # 沿Y方向扩展, 形成条状传感器阵列
    sensor_pos_arr = np.zeros((3, 0))
    offset = np.array([[0, 0.05, 0]]).T     # 条状传感器阵列中，传感器间距为50 mm
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

def fun_dipole_model(init_mag_prop, sensor_data):
    '''
    注意点: 
    1. 不要在该函数内更改传递参数sensor_data的值, 该函数会在迭代中循环调用, 
       假如这次更改了sensor_data值, 下次迭代会是变更过的值
    2. 
    '''
    # magnet_pos, magnet_ori, Bt = mag_prop
    a, b, c = init_mag_prop[0], init_mag_prop[1], init_mag_prop[2]
    m, n, p = init_mag_prop[3], init_mag_prop[4], init_mag_prop[5]
    Bt = init_mag_prop[6]
    Gx, Gy, Gz = init_mag_prop[7], init_mag_prop[8], init_mag_prop[9]

    # 预测环形传感器阵列的位置和方向
    beta1, beta2 = init_mag_prop[10:15], init_mag_prop[15:20]
    sensor_pos, sensor_ori = get_array_pose(beta1, beta2)

    # 将传感器测量值从传感器坐标系变换到全局坐标系 (B_g = R_gs * B_s)
    B_m = np.zeros_like(sensor_data)
    for i in range(sensor_ori.shape[0]):
        B_m[i] = np.dot(sensor_ori[i], sensor_data[i].T).T

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
    
    # 1-载入传感器采集的数据
    train_dataset = MagneticDatasetSeq(dir_path=cfg.dataset_dir, file_name=cfg.key_word, data_len=225, idx_offset=0)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2-第一个点的磁铁位姿初值 商用胶囊Bt=2.37e-8  N52 Bt=7.94e-08
    beta1 = np.array([60.0,  90.0,  120.0, 150, 180])
    beta2 = np.array([-60.0,  -90.0, -120.0, -150, -180])
    init_mag_prop = np.array([-0.025, 0.115, 0.09, 0, 0, 1, 2.37e-08, 0, 0, 0])
    init_mag_prop = np.append(init_mag_prop, np.concatenate((beta1, beta2)))

    est_mag_prop_arr = np.zeros((0, init_mag_prop.shape[0]))
    for data_sensor in train_loader:
        B_m = data_sensor.numpy().reshape(-1, 3)*1e-6   # uT -> T (real)

        # 阈值判定
        if(np.sum(np.abs(B_m))*1e6 < 500):
            print("continue")
            continue

        # 通过最小二乘法计算位姿
        tic = timer()
        res_lsq = least_squares(fun_dipole_model, init_mag_prop, method="lm", args=(B_m,))
        print("cost time: %f (s)" % (timer() - tic))

        # 结果处理
        if not res_lsq.success:
            print("success: ", res_lsq.success)
            exit()
        est_mag_prop = res_lsq.x
        init_mag_prop = est_mag_prop            # 结果作为下一次迭代的初值
        print(est_mag_prop)

        # 添加到列表中
        est_mag_prop_arr = np.append(est_mag_prop_arr, est_mag_prop.reshape(1, -1), axis=0)
    
    return est_mag_prop_arr


def args_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/volunteer3')
    parser.add_argument('--key_word', type=str, default="data_sequ_120919_nogeo.txt")
    parser.add_argument('--save_dir', type=str, default="./results/")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = args_cfg()

    est_mag_pose = pose_estimate(cfg)
    file_save_name = cfg.key_word.replace("data", "flexible_pose")
    np.savetxt(os.path.join(cfg.save_dir, file_save_name), est_mag_pose, delimiter=",")

    # 测试euler与mnp之间的转换
    # alpha, beta = -130, -90
    # m, n, p = euler_to_mnp(alpha, beta, degrees=True)
    # a, b = mnp_to_euler(m, n, p, degrees=True)
    # print(a, b)