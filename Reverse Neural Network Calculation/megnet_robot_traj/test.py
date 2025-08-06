import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP_Att import MLPWithAttention
from dataset.magnet_dataset import MGDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.eval_utils import draw_curve_test
import os


def to_tensor(input_val):
    return [torch.tensor(i).unsqueeze(0) for i in input_val]



def test():

    cs = 0.8
    E = [10, 15, 20]
    L = [10, 10, 100.8]
    mt = 10

    input_dim = 10
    hidden_dim = 20
    output_dim = 3
    
    E = to_tensor(E)
    L = to_tensor(L)

    input_batch = {}
    input_batch['cs'] = torch.tensor(cs).unsqueeze(0)
    input_batch['E'] = E
    input_batch['L'] = L
    input_batch['mt'] = torch.tensor(mt).unsqueeze(0)

    model = MLPWithAttention(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(os.path.join('results', 'model.pt'), weights_only=True))
    model.eval()

    result = model(input_batch)
    draw_curve_test(
        result, f"cs{cs:.2f}_E{E[0].item()}-{E[1].item()}-{E[2].item()}_L{L[0].item()}-{L[1].item()}-{L[2].item():.2f}_{mt}mT")


if __name__ == '__main__':
    test()
