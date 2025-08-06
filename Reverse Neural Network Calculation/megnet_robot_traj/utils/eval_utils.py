from dataset.magnet_dataset import quadratic_curve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_curve(params, gt_params, set_id):
    a, b, c = params
    print(set_id, a, b, c)
    gt_a, gt_b, gt_c = gt_params
    x = np.linspace(0, 40, 400)
    y = a*(x**2) + b*x + c
    gt_y = gt_a*(x**2) + gt_b*x + gt_c
    
    plot_gt_estimate(x, gt_y, gt_a, x, y, a, filename=f"{set_id}.png")
    
def draw_curve_new(params, gt_params, set_id):
    a = params
    gt_a = gt_params
    print(set_id, a, gt_a)

    x = np.linspace(0, 40, 400)
    y = a*(x**2)
    gt_y = gt_a*(x**2)
    
    plot_gt_estimate(x, gt_y, gt_a, x, y, a, filename=f"{set_id}.png")     
    
def plot_gt_estimate(gt_x, gt_y, gt_a, est_x, est_y, est_a, filename='gt_estimate_plot.png'):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # Plot ground truth on the left subplot
    ax1.plot(gt_x, gt_y, label='Ground Truth', color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.title(f'GT Curve: y = {gt_a:.6f}x^2\nEST Curve: y = {est_a:.6f}x^2')
    # ax1.set_xlim(0, 40)  # Set x-axis range from 2 to 8
    # ax1.set_ylim(0, 400)  # Set y-axis range from -0.5 to 0.5

    # Plot estimate on the right subplot
    ax1.plot(est_x, est_y, label='Estimate', color='red')
    ax1.legend()

    # Save the figure
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    
def draw_curve_test(params, set_id):
    a, b, c = params[:, 0].item(), params[:, 1].item(), params[:, 2].item()
    x = np.linspace(0, 40, 400)
    y = a*(x**2) + b*x + c
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    plt.plot(x, y, label='Testing', color='blue')
    plt.title('Testing')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{set_id}.png"))
    plt.close()
    