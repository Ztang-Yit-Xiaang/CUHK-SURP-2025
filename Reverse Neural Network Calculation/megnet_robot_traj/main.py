import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP_Att import MLPWithAttention
from dataset.magnet_dataset import MGDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.eval_utils import draw_curve, draw_curve_new
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from inverse_optimize import batch_inverse_optimize_varying_mt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(42)


def main(config):
    data = torch.randn(1000, 10)  # 1000 samples, each with 10 features
    gt_params = torch.randint(0, 2, (1000,))  # 1000 gt_params (binary classification)
    data_dir = 'Sim_Magenet'

    # Create datasets and dataloaders
    train_dataset = MGDataset(data_dir=data_dir,  mode='train')
    val_dataset = MGDataset(data_dir=data_dir, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    model = MLPWithAttention(input_dim, hidden_dim, output_dim)
    criterion = nn.L1Loss()
    criterion_test = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1000*len(train_loader), gamma=0.1)
    
    # Training loop
    num_epochs = 2500
    pbar = tqdm(range(num_epochs))
    writer = SummaryWriter('tf')
    for epoch in range(num_epochs):
        model.train()
        losses = 0
        for batch in train_loader:
            gt_params = batch['params'].float()
            optimizer.zero_grad()
            outputs = model(batch)
            if outputs.ndim==1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(gt_params, outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
        writer.add_scalar('loss', losses/len(train_loader), epoch+1)
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses/len(train_loader)}')
        pbar.update(1)
    pbar.close()
    writer.close()
    # Test
    model.eval()
    val_outputs = []
    val_gt_params_list = []
    with torch.no_grad():
        for batch in val_loader:
            gt_params = batch['params'].float()
            outputs = model(batch)
            if outputs.ndim==1:
                outputs = outputs.unsqueeze(1)
            val_outputs.append(outputs)
            val_gt_params_list.append(gt_params)
            # draw_curve((outputs[:, 0].item(), outputs[:, 1].item(), outputs[:, 2].item()), 
            #            (gt_params[:, 0].item(), gt_params[:, 1].item(), gt_params[:, 2].item()),
            #            batch['id'][0].replace('(base)', '').replace('/', '').replace('.txt', ''))
            draw_curve_new(outputs[:, 0].item(), gt_params[:, 0].item(), 
                           batch['id'][0].replace('(base)', '_').replace("\\", '').replace('.txt', ''))
    val_gt = torch.stack(val_gt_params_list)
    val_out = torch.stack(val_outputs)
    test_loss = criterion_test(val_gt, val_out)
    
    torch.save(model.state_dict(), os.path.join('results', 'model.pt'))
    print(f"Test MSE Loss: {test_loss}")
    os.makedirs("Inverse_Results/csv", exist_ok=True)
    os.makedirs("Inverse_Results/Plot", exist_ok=True)
    print("\n Running inverse optimization...")
    batch_inverse_optimize_varying_mt(a_start=0.002, a_end=0.0024, a_step=0.0001)
    print(" Inverse optimization complete.")

if __name__ == '__main__':
    main(None)