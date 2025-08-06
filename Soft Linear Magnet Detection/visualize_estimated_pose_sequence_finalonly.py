import numpy as np
from simulation.pose_estimate_improved import fun_dipole_model
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
class ArgsCfg: pass
def args_cfg():
    cfg = ArgsCfg()
    cfg.dataset_dir = './dataset'
    cfg.key_word = 'captured_20250728_134404.npy'
    return cfg

def estimate_one_frame(sensor_data, init_mag_prop, betas, frame_idx, cfg):
    # 合理範圍設定
    lower_bounds = np.array([
        -0.2, -0.2, -0.1,   # 位置
        -2, -2, -2,         # 方向
        -5e-2,              # Bt
        -1e-4, -1e-4, -1e-4 # 地磁
    ])  # 固定 beta1/2
    lower_bounds = np.concatenate((lower_bounds, betas-10))
    upper_bounds = np.array([
        0.2, 0.0, 0.15,      # 位置
        2, 2, 2,            # 方向
        5e-2,               # Bt
        1e-4, 1e-4, 1e-4    # 地磁
    ])  # 固定 beta1/2 這裡設上下限一樣即可
    upper_bounds = np.concatenate((upper_bounds, betas+10))
    res = least_squares(fun_dipole_model, init_mag_prop, method='trf', bounds=(lower_bounds, upper_bounds), args=(sensor_data, frame_idx, cfg), max_nfev=400)
    est_mag_prop = res.x
    B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori = fun_dipole_model(
        est_mag_prop, sensor_data, return_full=True, frame_idx=frame_idx, cfg=cfg)
    return est_mag_prop, B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori

def visualize_estimation(sensor_pos, sensor_ori, dipole_pos, dipole_ori, B_pred, title=""):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2], color='blue', label='Sensor Positions')
    ax.scatter(dipole_pos[:, 0], dipole_pos[:, 1], dipole_pos[:, 2], color='red', label='Dipole Centers')
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
    for i in range(sensor_pos.shape[0]):
        ax.quiver(sensor_pos[i, 0], sensor_pos[i, 1], sensor_pos[i, 2],
                  B_pred[i, 0], B_pred[i, 1], B_pred[i, 2],
                  length=0.03, color='green', normalize=True)
        ax.quiver(sensor_pos[i, 0], sensor_pos[i, 1], sensor_pos[i, 2],
                  sensor_ori[i, 0, 2], sensor_ori[i, 1, 2], sensor_ori[i, 2, 2],
                  length=0.02, color='cyan', normalize=True)
    ax.set_xlim((-0.25, 0.25))
    ax.set_ylim((-0.3, 0.05))
    ax.set_zlim((-0.1, 0.15))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    if title:
        ax.set_title(title)
    plt.show()

# ---- 主程式 ----
if __name__ == "__main__":
    data = np.load('./dataset/captured_20250731_161948.npy')
    cfg = args_cfg()
    start_frame, end_frame = 70, 100
    prev_mag_prop = None

    final_sensor_data = None
    final_est_mag_prop = None
    final_sensor_pos = None
    final_sensor_ori = None
    final_dipole_pos = None
    final_dipole_ori = None
    final_B_pred = None

    fixed_beta1 = np.array([65,  90,  120,  160, 180]) + np.array([0, 2, -3, 1, -2])
    fixed_beta2 = np.array([-50,  -90,  -125,  -150, -180]) + np.array([1, -1, 2, -3, 0])
    betas = np.concatenate((fixed_beta1, fixed_beta2))
    for frame_idx in range(start_frame, end_frame):
        sensor_data = data[frame_idx].reshape(85, 3)[:75]
        if prev_mag_prop is None:
            init_mag_prop = np.array([0.02, -0.01, -0.04, 0.10, -0.30, 0.10, 0.0, 5e-6, 5e-6, 5e-6])
            init_mag_prop = np.concatenate((init_mag_prop, fixed_beta1, fixed_beta2))
        else:
            init_mag_prop = prev_mag_prop.copy()

        est_mag_prop, B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori = estimate_one_frame(
            sensor_data, init_mag_prop, betas, frame_idx, cfg)
        prev_mag_prop = est_mag_prop.copy()

        final_sensor_data = sensor_data
        final_est_mag_prop = est_mag_prop
        final_sensor_pos = sensor_pos
        final_sensor_ori = sensor_ori
        final_dipole_pos = dipole_pos
        final_dipole_ori = dipole_ori
        final_B_pred = B_pred

    print(f"\n=== Final Frame {end_frame} ===")
    print("dipole_ori = \n", final_dipole_ori)
    print("sensor_data shape:", final_sensor_data.shape)
    print("sensor_pos shape:", final_sensor_pos.shape)
    print("sensor_ori shape:", final_sensor_ori.shape)
    print("dipole_pos shape:", final_dipole_pos.shape)
    print("B_pred shape:", final_B_pred.shape)

    visualize_estimation(final_sensor_pos, final_sensor_ori, final_dipole_pos, final_dipole_ori, final_B_pred,
                         title=f"Pose Estimation Final Frame {end_frame}")