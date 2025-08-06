import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure
from scipy.optimize import curve_fit
import re


def quadratic_curve(x, a, b, c):
    return a * x**2 + b * x + c

def cal_curvature(x, a, b):
    numer = np.abs(2*a)
    deno = np.sqrt(np.power((1+2*a*x+b), 3))
    return numer/deno

def draw_curve(x, y):
    params, _ = curve_fit(quadratic_curve, x, y)
    a, b, c = params
    x = np.linspace(0, 40, 400)
    y = a*(x**2) + b*x + c
    plt.plot(x, y)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Quadratic Curve: y = {a}x^2+{b}x+{c}')
    plt.savefig('quadratic_curve.png')
    plt.show()
    
def read_txt(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        line_list = list(f.readlines())
        for i, line in enumerate(line_list):
            if i < 8:
                continue
            line_data = re.split(r'\s+', line)
            line_data = [float(j) for j in line_data[:7]]
            data.append(line_data)
    return data


if __name__ == '__main__':
    file_path = 'F:/UMN Researches/CUHK Research/megnet_robot_traj/megnet_robot_traj/Sim_Magenet/cs0.8_E10-15-20_L10-10-10(base)/0.8_10mT.txt'
    data = np.array(read_txt(file_path))
    x = data[:, 0]
    y = data[:, 2]
    draw_curve(x, y)