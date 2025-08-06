# ✅ 文件：ring_array_model.py（原始環形模型文件中擴展）
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_magnet_pose(betas, segment_length=0.07):
    """
    將撓度角序列 beta（單位°）轉換為磁鐵上的多個 dipole 的位置與方向。

    參數：
    - betas: list or array of bending angles β（度）
    - segment_length: 每段磁鐵的長度（預設 7cm）

    返回：
    - dipole_pos_all: (N, 3) 多個 dipole 的空間位置
    - dipole_ori_all: (N, 3) 多個 dipole 的方向單位向量
    """
    dipole_pos_all = []
    dipole_ori_all = []

    current_pos = np.array([[0], [0], [0]])   # 初始點 (3,1)
    current_ori = np.eye(3)                   # 初始朝向為X軸

    for beta in betas:
        R_y = R.from_euler('y', beta, degrees=True).as_matrix()  # (3,3)
        current_ori = current_ori @ R_y
        next_pos = current_pos + current_ori @ np.array([[segment_length], [0], [0]])

        dipole_pos_all.append(next_pos.ravel())
        dipole_ori_all.append(current_ori[:, 0])  # X軸方向作爲 dipole 朝向

        current_pos = next_pos

    return np.array(dipole_pos_all), np.array(dipole_ori_all)
