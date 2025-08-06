import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline


def visualize_estimation (sensor_pos, sensor_ori, dipole_pos, dipole_ori, B_pred = None, ax = None, title=""):
    print("sensor_pos.shape =", sensor_pos.shape)
    print("sensor_pos[0:5] =", sensor_pos[0:5])
    print("sensor_ori[0:2] =", sensor_ori[0:2])
    print("dipole_pos.shape =", dipole_pos.shape)
    print("dipole_ori.shape =", dipole_ori.shape)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # 绘制传感器位置（蓝点）
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
               color='blue', label='Sensor Positions')
    # 插值平滑曲线连接 dipole_pos

    x, y, z = dipole_pos[:, 0], dipole_pos[:, 1], dipole_pos[:, 2]
    t = np.linspace(0, 1, len(dipole_pos))
    spl_x = make_interp_spline(t, x, k=3)
    spl_y = make_interp_spline(t, y, k=3)
    spl_z = make_interp_spline(t, z, k=3)

    t_new = np.linspace(0, 1, 200)
    x_new = spl_x(t_new)
    y_new = spl_y(t_new)
    z_new = spl_z(t_new)

    ax.plot(x_new, y_new, z_new, color='orange', linewidth=2, label='Fitted Magnet Curve')

    # 绘制 dipole 位置（红点）与方向（红箭头）
    ax.scatter(dipole_pos[:, 0], dipole_pos[:, 1], dipole_pos[:, 2],
               color='red', label='Dipole Centers', s=50)
    
    # for i in range(dipole_pos.shape[0]):
    #     ax.quiver(dipole_pos[i, 0], dipole_pos[i, 1], dipole_pos[i, 2],
    #               dipole_ori[i, 0], dipole_ori[i, 1], dipole_ori[i, 2],
    #               length=0.05, color='red', normalize=True)
    if B_pred is not None:
        # 绘制预测磁场方向（绿箭头）
        for i in range(sensor_pos.shape[0]):
            ax.quiver(sensor_pos[i, 0], sensor_pos[i, 1], sensor_pos[i, 2],
                      B_pred[i, 0], B_pred[i, 1], B_pred[i, 2],
                    length=0.03, color='green', normalize=True)
        # 绘制传感器方向（青箭头）
        for i in range(sensor_pos.shape[0]):
            ax.quiver(sensor_pos[i, 0], sensor_pos[i, 1], sensor_pos[i, 2],
                    sensor_ori[i, 0, 2],  # x component of Z axis
                    sensor_ori[i, 1, 2],  # y component of Z axis
                    sensor_ori[i, 2, 2],  # z component of Z axis
                    length=0.02, color='cyan', normalize=True)
        for i, pos in enumerate(sensor_pos):
            ax.text(pos[0], pos[1], pos[2], str(i), color='black')


    # 畫全局三軸
    ax.quiver(0, 0, 0, -0.1, 0, 0, color='r', label='X-axis')
    ax.quiver(0, 0, 0, 0, -0.1, 0, color='g', label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', label='Z-axis')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from simulation.pose_estimate_improved import fun_dipole_model
    from scipy.optimize import least_squares
    data = np.load('./dataset/captured_20250728_134404.npy')
    prev_mag_prop = None
    start_frame = 50
    end_frame = 110  # 可調整範圍
    for frame_idx in range(start_frame, end_frame):
        sensor_data = data[frame_idx]
        sensor_data = sensor_data.reshape(85, 3)[:75]  # (75,3)
        if prev_mag_prop is None:
            init_mag_prop = np.array([
                0.02, -0.01, -0.04,     # 磁鐵位置
                1.00, -0.30, 1.00,      # 磁鐵方向
                0.0,                    # Bt
                5e-6, 5e-6, 5e-6        # 地磁偏移
            ])
            beta1 = np.array([65,  90,  120,  160, 180]) + np.array([0, 2, -3, 1, -2])
            beta2 = np.array([-50,  -90,  -125,  -150, -180]) + np.array([1, -1, 2, -3, 0])
            init_mag_prop = np.concatenate((init_mag_prop, beta1, beta2))
        else:
            init_mag_prop = prev_mag_prop.copy()
            res = least_squares(fun_dipole_model, init_mag_prop, method='lm', args=(sensor_data, frame_idx))
            est_mag_prop = res.x
            prev_mag_prop = est_mag_prop.copy()
    B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori = fun_dipole_model(est_mag_prop, sensor_data, return_full=True, frame_idx=frame_idx)
    # 1. 加载配置
    # Define a minimal args_cfg function to avoid NameError
    class ArgsCfg:
        pass
    def args_cfg():
        return ArgsCfg()
    cfg = args_cfg()
    cfg.dataset_dir = './dataset'
    cfg.key_word = 'captured_20250728_134404.npy'

    # 2. 执行姿态估计，获取磁铁参数
    # est_mag_prop_arr = pose_estimate(cfg)
    # if est_mag_prop_arr is None or est_mag_prop_arr.shape[0] == 0:
    #     print("❌ 无估计结果")
    #     exit()
    print("dipole_ori = \n", dipole_ori)

    print(f"\n=== Frame {frame_idx} ===")
    print("dipole_ori = \n", dipole_ori)
    print("sensor_data shape:", sensor_data.shape)

    # 4. 打印结果
    print("sensor_pos shape:", sensor_pos.shape)
    print("sensor_ori shape:", sensor_ori.shape)
    print("dipole_pos shape:", dipole_pos.shape)
    print("B_pred shape:", B_pred.shape)



    # 5. 可视化
    visualize_estimation(sensor_pos, sensor_ori, dipole_pos, dipole_ori, B_pred,
                         title="Pose Estimation: Magnetic Field Visualization")
