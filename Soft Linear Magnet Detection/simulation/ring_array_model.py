'''
传感器阵列建模
输入: 单元的长度和角度（夹角）
输出: 传感器的3维位置和姿态
'''
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

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
    offset = np.array([[0, 0, 0.05]]).T     # 条状传感器阵列中，传感器间距为50 mm
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*2, axis=1)   # 从上到下
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
    p_to_hinge = np.array([[-h12_len, 0, 0]]).T              # 指向下一个铰链的向量
    p_to_sensor = np.array([[-h12_len/2, 0, s_height]]).T    # 指向传感器的向量
    theta = [0, 0, h1_beta]            # 沿Z轴旋转
    R_12 = R.from_euler('XYZ', theta, degrees=True)

    # 计算铰链2的位置
    h2_pos = h1_pos + R_12.as_matrix().dot(p_to_hinge)

    # 计算传感器的位置
    sensor_pos = h1_pos + R_12.as_matrix().dot(p_to_sensor)

    # 沿Z方向扩展, 形成条状传感器阵列
    sensor_pos_arr = np.zeros((3, 0))
    offset = np.array([[0, 0, 0.05]]).T     # 条状传感器阵列中，传感器间距为50 mm
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*2, axis=1)   # 从上到下
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos + offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*0, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*1, axis=1)
    sensor_pos_arr = np.append(sensor_pos_arr, sensor_pos - offset*2, axis=1)
    
    # 姿态扩展, 采用expand_dims在0维度上增加一个维度, 再在0维度上重复5次
    sensor_ori_arr = np.expand_dims(R_12.as_matrix(), 0).repeat(5, axis=0)

    return sensor_pos_arr.T, sensor_ori_arr, h2_pos

def get_linear_flexible_array_pose(start_pos, start_dir, bend_angles, length = 0.3, num_segments = 15):
    """
    構造一條線性柔性磁鐵的分段 dipole 分佈：
    - start_pos: 初始磁鐵位置 (3,)
    - start_dir: 初始方向單位向量 (3,)
    - length: 磁鐵總長度 (m)
    - num_segments: 段數 N （若磁鐵總長約0.25m，則 N=10~16）
    - bend_angles: 撓曲角度 (N 個，角度制)

    返回：每段中心的 sensor_pos_all (N, 3)，和方向 sensor_ori_all (N, 3)
    """
    segment_len = length / num_segments # 離散化每段的長度
    # 檢查 bend_angles 的類型
    if bend_angles is None:
        bend_angles = [0] * num_segments
    elif isinstance(bend_angles, (int, float)):
        bend_angles = [bend_angles] * num_segments
    elif isinstance(bend_angles, list):
        bend_angles = np.array(bend_angles)
    elif isinstance(bend_angles, np.ndarray):
        if bend_angles.ndim == 0:
            bend_angles = np.array([bend_angles])
    else:
        raise TypeError("bend_angles 必須是 int, float, list 或 np.ndarray 類型")
    # 確保 bend_angles 的長度與 num_segments 相同
    if len(bend_angles) != num_segments:
        raise ValueError("bend_angles 的長度必須等於 num_segments")
    
    sensor_pos_all = []
    sensor_ori_all = []

    pos = np.array(start_pos).astype(np.float64).reshape(3)
    dir_vec = np.array(start_dir).astype(np.float64).reshape(3)
    dir_vec /= np.linalg.norm(dir_vec)

    for i in range(num_segments):
        # 添加當前段的中心點與方向
        center_pos = pos + dir_vec * (segment_len / 2) # 中心點位置
        sensor_pos_all.append(center_pos) # 保存中心位置
        sensor_ori_all.append(dir_vec.copy()) # 保存方向向量

        # 更新位置到下段起始點
        pos = pos + dir_vec * segment_len

        # 根據 bend_angle 更新方向
        if i < len(bend_angles):
            angle = np.deg2rad(bend_angles[i])
            Z_axis = np.array([0, 0, 1])  # Z 軸方向
            axis = np.cross(dir_vec, Z_axis)  # 計算旋轉軸
            if np.linalg.norm(axis) < 1e-6:  # 如果旋轉軸接近零向量，則不旋轉
                continue
            axis /= np.linalg.norm(axis)  # 標準化旋轉軸
            # 使用 scipy 的旋轉來更新方向   
            rot = R.from_rotvec(angle * axis)
            dir_vec = rot.apply(dir_vec)

    return np.array(sensor_pos_all), np.array(sensor_ori_all)

def get_spatial_flexible_array_pose(start_pos, start_dir, bend_angles, phi_angles=None, segment_len=0.01):
    """
    根據撓曲角與橫向角生成柔性 dipole 的空間位置與方向

    參數：
    - start_pos: 初始位置，形狀 (3,1)
    - start_dir: 初始方向，形狀 (3,1)，需為單位向量
    - bend_angles: 每段彎曲角，繞 Z 軸 (Yaw)
    - phi_angles: （可選）每段第二彎曲角，繞 Y 軸 (Pitch)
    - segment_len: 每段長度
    """
    num_segments = len(bend_angles)
    if phi_angles is None:
        phi_angles = np.zeros_like(bend_angles)

    pos_list = [start_pos.reshape(3)]
    dir_list = []

    # 初始旋轉方向
    R_global = R.align_vectors([start_dir.reshape(3)], [[1, 0, 0]])[0]  # 將 start_dir align 到 X 軸
    cur_pos = start_pos.reshape(3)

    for i in range(num_segments):
        # 每段彎曲 = Y → Z（先 Pitch, 再 Yaw）
        Ry = R.from_euler('y', phi_angles[i], degrees=True)
        Rz = R.from_euler('z', bend_angles[i], degrees=True)
        R_seg = Rz * Ry
        R_global = R_global * R_seg

        # 更新方向與位置
        dir_vec = R_global.apply([1, 0, 0])  # local x 軸方向
        center_pos = cur_pos + 0.5 * segment_len * dir_vec
        pos_list.append(center_pos)
        dir_list.append(dir_vec)

        cur_pos = cur_pos + segment_len * dir_vec

    pos_arr = np.array(pos_list[:-1])
    dir_arr = np.array(dir_list)
    return pos_arr, dir_arr

if __name__ == "__main__":
    
    main_unit_width = 0.05      # 每个单元的宽度为50mm (两个铰链之间的长度)
    slave_unit_width = 0.07     # 子板的宽度
    sensor_height = -0.01       # 传感器离铰链平面的高度 (负号表示沿Z轴负方向)
    sensor_pos_all = np.zeros((0, 3))
    sensor_ori_all = np.zeros((0, 3, 3))

    # 主控板(基板)
    sensor_pos_base, sensor_ori_base, hinge_pos_base = \
        ring_array_pos_ext(np.array([[0, 0, 0]]).T, 0, 0, sensor_height)
    sensor_pos_all = np.append(sensor_pos_all, sensor_pos_base, axis=0)
    sensor_ori_all = np.append(sensor_ori_all, sensor_ori_base, axis=0)
    print("sensor_pos_base: \n", sensor_pos_base)
    print("sensor_ori_base: \n", sensor_ori_base)

    
    # 沿X负方向扩展
    # beta1 = np.array([45, 90, 90, 120, 150, 180])
    beta1 = np.array([90, 90, 90, 180, 180, 180])
    _, _, hinge_pi_pos = ring_array_neg_ext(hinge_pos_base, 0, main_unit_width/2, sensor_height)       # 注意/2, sensor_height前面的负号
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
    # beta2 = np.array([-45,  -90, -90, -120, -150, -180])
    beta2 = np.array([-90,  -90, -90, -90, -90, -180])
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

    print("sensor_pos_all: \n", sensor_pos_all)
    print("sensor_ori_all: \n", sensor_ori_all)

    np.save("./results/sensor_pos.npy", sensor_pos_all)
    np.save("./results/sensor_ori.npy", sensor_ori_all)
