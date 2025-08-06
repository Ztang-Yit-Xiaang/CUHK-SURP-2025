'''
根据一组匹配好的3D点对, 估计两个坐标系间的位姿关系
采用欧拉角(alpha, beta, gamma)表示姿态
- 3个角都作为优化参数进行优化时, 在beta为90°时(Gimbal Lock), alpha和gamma不会是唯一值
- 只有beta为优化参数时, 即时beta为90, 也能优化出正确结果出来
'''
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def data_generate(alpha, beta, gamma):
    # 随机生成坐标系2下的点
    p2_arr = np.random.rand(5, 3)*10 - 5

    # 坐标系2相对于坐标系1的位姿
    euler_angle = [alpha, beta, gamma]            # 沿Y轴旋转
    R_12 = R.from_euler('XYZ', euler_angle, degrees=True)

    # 将坐标系2的点变换到坐标系1下的点
    p1_arr = np.zeros((0, 3))
    for pi in p2_arr:
        p1 = R_12.as_matrix().dot(np.expand_dims(pi, axis=0).T)
        p1_arr = np.append(p1_arr, p1.T, axis=0)
    
    print(p2_arr)
    print(p1_arr)
    return p1_arr, p2_arr


def residual_error(x, p1_arr, p2_arr):
    alpha, beta, gamma = x[0], x[1], x[2]       # 3个优化参数
    # alpha, beta, gamma = 0, x[0], 0           # 只有1个优化参数

    euler_angle = [alpha, beta, gamma]
    R_12 = R.from_euler('XYZ', euler_angle, degrees=True)

    p1_arr_temp = np.zeros((0, 3))
    for pi in p2_arr:
        p1 = R_12.as_matrix().dot(np.expand_dims(pi, axis=0).T)
        p1_arr_temp = np.append(p1_arr_temp, p1.T, axis=0)

    p1_arr_temp = p1_arr_temp.ravel()
    p1_arr = p1_arr.ravel()
    error_arr = p1_arr - p1_arr_temp

    return error_arr


if __name__ == "__main__":
    alpha, beta, gamma = 0, 70, 0
    p1_arr, p2_arr = data_generate(alpha, beta, gamma)

    x0 = [0, 40, 0]          # 初始值
    res_lsq = least_squares(residual_error, x0, method="lm", args=(p1_arr, p2_arr))
    if not res_lsq.success:
        print("success: ", res_lsq.success)
    else:
        print("result: ", res_lsq.x)

