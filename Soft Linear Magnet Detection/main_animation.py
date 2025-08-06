import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation.pose_estimate_improved import pose_estimate_single, args_cfg, fun_dipole_model
from visualize_estimated_pose import visualize_estimation
cfg = args_cfg()
cfg.dataset_dir = './dataset'
cfg.key_word = 'captured_data_20250724_173504'
# 读取数据
data_all = np.load('./dataset/captured_data_20250724_173504.npy')  # shape = (T, 85, 3)
num_frames = data_all.shape[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame_idx):
    ax.clear()
    B_pred, sensor_pos, sensor_ori, dipole_pos, dipole_ori = pose_estimate_single(cfg, frame_idx)
    artists = visualize_estimation(sensor_pos, sensor_ori, dipole_pos, dipole_ori,
                                  B_pred, ax=ax, title=f"Frame {frame_idx}")
    ax.set_xlim((-0.25, 0.25))
    ax.set_ylim((-0.25, 0.25))
    ax.set_zlim((-0.1, 0.3))
    # Ensure artists is iterable, or return [ax] as a fallback
    if artists is not None:
        return artists if hasattr(artists, '__iter__') else [artists]
    return [ax]

ani = FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=True)
plt.show()
