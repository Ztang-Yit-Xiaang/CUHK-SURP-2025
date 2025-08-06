import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("./")

def show_pos(x, y, z):
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    # ax.scatter(x[0], y[0], z[0], c='r')
    # ax.scatter(x[0:111], y[0:111], z[0:111])
    # ax.scatter(x[112], y[112], z[112], c='r')
    # ax.scatter(x[112:], y[112:], z[112:], c='g')
    ax1.scatter(x[0], y[0], z[0], s=10, c='r')
    ax1.scatter(x, y, z, s=10)

    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    plt.title('3D Scatter Plot')
    # plt.show()

def show_geo(x, Gx, Gy, Gz):
    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(1, 3, 1)
    ax1.plot(x, Gx, 'r', label='Data 1')  # 红色曲线

    ax2 = fig2.add_subplot(1, 3, 2)
    ax2.plot(x, Gy, 'g', label='Data 2')  # 绿色曲线

    ax3 = fig2.add_subplot(1, 3, 3)
    ax3.plot(x, Gz, 'b', label='Data 3')  # 蓝色曲线
    
    # plt.show()

def show_beta(x, beta1, beta2):
    fig3 = plt.figure(3)
    ax31 = fig3.add_subplot(5, 1, 1)
    ax31.plot(x, beta1[:, 0], 'r', label='Data 1')  # 红色曲线
    ax32 = fig3.add_subplot(5, 1, 2)
    ax32.plot(x, beta1[:, 1], 'g', label='Data 2')  # 绿色曲线
    ax33 = fig3.add_subplot(5, 1, 3)
    ax33.plot(x, beta1[:, 2], 'b', label='Data 3')  # 蓝色曲线
    ax34 = fig3.add_subplot(5, 1, 4)
    ax34.plot(x, beta1[:, 3], 'r', label='Data 1')  # 红色曲线
    ax35 = fig3.add_subplot(5, 1, 5)
    ax35.plot(x, beta1[:, 4], 'g', label='Data 2')  # 绿色曲线

    fig4 = plt.figure(4)
    ax41 = fig4.add_subplot(5, 1, 1)
    ax41.plot(x, beta2[:, 0], 'r', label='Data 1')  # 红色曲线
    ax42 = fig4.add_subplot(5, 1, 2)
    ax42.plot(x, beta2[:, 1], 'g', label='Data 2')  # 绿色曲线
    ax43 = fig4.add_subplot(5, 1, 3)
    ax43.plot(x, beta2[:, 2], 'b', label='Data 3')  # 蓝色曲线
    ax44 = fig4.add_subplot(5, 1, 4)
    ax44.plot(x, beta2[:, 3], 'r', label='Data 1')  # 红色曲线
    ax45 = fig4.add_subplot(5, 1, 5)
    ax45.plot(x, beta2[:, 4], 'g', label='Data 2')  # 绿色曲线
    # plt.show()

if __name__ == "__main__":
    # 载入数据
    raw_data = np.loadtxt("./results/volunteer4/fix_pose_sequ_130843_8300~end.txt", dtype=str, delimiter=",")
    data = raw_data[:, 1:].astype(float)        # 去除时间戳

    # 1-绘制3D轨迹图
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    show_pos(x, y, z)


    # 2-绘制地磁变化图
    Gx, Gy, Gz = data[:, 7], data[:, 8], data[:, 9]
    x = range(len(Gx))
    show_geo(x, Gx, Gy, Gz)


    # 3-阵列角度绘制
    # beta1, beta2 = data[:, 10:15], data[:, 15:]
    # show_beta(x, beta1, beta2)

    plt.show()
