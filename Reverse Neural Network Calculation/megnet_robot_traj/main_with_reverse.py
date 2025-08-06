
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP_Att import MLPWithAttention
from dataset.magnet_dataset import MGDataset
from torch.utils.data import DataLoader
from utils.eval_utils import draw_curve_new
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

from reverse_search import reverse_search

def main(config):
    data_dir = 'Sim_Magenet'

    train_dataset = MGDataset(data_dir=data_dir, mode='train')
    val_dataset = MGDataset(data_dir=data_dir, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    model = MLPWithAttention(input_dim, hidden_dim, output_dim)
    criterion = nn.L1Loss()
    criterion_test = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1000 * len(train_loader), gamma=0.1)

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
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            loss = criterion(gt_params, outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
        writer.add_scalar('loss', losses / len(train_loader), epoch + 1)
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses / len(train_loader):.6f}')
        pbar.update(1)
    pbar.close()
    writer.close()

    model.eval()
    val_outputs = []
    val_gt_params_list = []
    with torch.no_grad():
        for batch in val_loader:
            gt_params = batch['params'].float()
            outputs = model(batch)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            val_outputs.append(outputs)
            val_gt_params_list.append(gt_params)
            draw_curve_new(outputs[:, 0].item(), gt_params[:, 0].item(),
                           batch['id'][0].replace('(base)', '_').replace('\\', '').replace('.txt', ''))
    val_gt = torch.stack(val_gt_params_list)
    val_out = torch.stack(val_outputs)
    test_loss = criterion_test(val_gt, val_out)

    torch.save(model.state_dict(), os.path.join('results', 'model.pt'))
    print(f"Test MSE Loss: {test_loss:.6f}")

    # === Reverse search integration ===
    print("\n Starting Reverse Search:")
    target_a = 0.00235
    matches = reverse_search(target_a)
    print(f"\n Found {len(matches)} matching configurations for a_hat ≈ {target_a}:")
    for i, (mt, E, L, cs, pred) in enumerate(matches):
        print(f"{i+1:>2}. mt={mt}, E={E}, L={L}, cs={cs:.2f} → a_hat_pred={pred:.6f}")


if __name__ == '__main__':
    main(None)
