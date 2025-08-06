from scipy.optimize import least_squares
from pose_estimate_improved import fun_dipole_model, get_array_pose, calc_dipole_field
import numpy as np
import matplotlib.pyplot as plt

# ====================== 測試設置 =========================
np.random.seed(42)  # 為了結果可重現

# 1. 設定 Ground Truth 磁鐵參數
gt_pos = np.array([0.02, 0.01, 0.06])      # 磁鐵中心位置 (m)
gt_ori = np.array([0.0, 0.0, 1.0])         # 磁鐵朝向 z 軸
Bt = 8e-8                                  # 磁強度 (T)
Geo = np.array([0.0, 0.0, 0.0])            # 地磁偏移

# 2. 模擬彎曲角度（單位: degree）
beta1 = np.array([5, 5, 5, 5, 5, 5])
beta2 = np.array([-5, -5, -5, -5, -5, -5])

# 3. 生成感測器位置與姿態
sensor_pos, sensor_ori = get_array_pose(np.zeros(6), np.zeros(6))  # 平直排布

# 4. 生成模擬測量磁場 B_measured
B_measured = np.zeros((sensor_pos.shape[0], 3))
magnet_pos_list, magnet_ori_list = get_array_pose(beta1, beta2)
for i in range(sensor_pos.shape[0]):
    B_total = np.zeros(3)
    for j in range(len(magnet_pos_list)):
        B_total += calc_dipole_field(sensor_pos[i], magnet_pos_list[j], magnet_ori_list[j], Bt / len(magnet_pos_list))
    B_measured[i] = B_total + Geo

# 5. 初始猜測值（添加小偏差）
init_pos = gt_pos + 1e-3*np.random.randn(3)
init_ori = gt_ori + 0.1*np.random.randn(3)
init_ori /= np.linalg.norm(init_ori)
init_prop = np.concatenate([
    init_pos,
    init_ori,
    np.array([Bt]),
    Geo,
    beta1,
    beta2
])

# ====================== 執行估計 =========================
res = least_squares(fun_dipole_model, init_prop,
                    method='lm',
                    args=(sensor_pos, sensor_ori, B_measured))

est_prop = res.x

# ====================== 結果輸出 =========================
print("=== Estimated Result ===")
print("Position (m):", est_prop[:3])
print("Orientation (unit vector):", est_prop[3:6])
print("Bt (T):", est_prop[6])
print("Geo Bias:", est_prop[7:10])

# 誤差分析
pos_error = np.linalg.norm(est_prop[0:3] - gt_pos) * 1000  # mm
ori_error = np.arccos(np.clip(np.dot(est_prop[3:6], gt_ori), -1.0, 1.0)) * 180 / np.pi
print("位置誤差: %.2f mm" % pos_error)
print("方向誤差: %.2f 度" % ori_error)

# 可視化磁場預測 vs 測量（選擇一個 sensor）
plt.figure()
plt.plot(B_measured[:, 0], label='Measured Bx')
plt.plot(B_measured[:, 1], label='Measured By')
plt.plot(B_measured[:, 2], label='Measured Bz')
plt.title("模擬 B_measured (真實磁場)")
plt.legend()
plt.xlabel("Sensor Index")
plt.ylabel("Magnetic Field (T)")
plt.grid()
plt.tight_layout()
plt.show()
