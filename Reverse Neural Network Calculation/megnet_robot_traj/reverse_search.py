
import torch
from model.MLP_Att import MLPWithAttention
import itertools

def reverse_search(target_a_hat, tolerance=1e-4):

    # Initialize and load model
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    model = MLPWithAttention(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('results/model.pt'))
    model.eval()

    # Define search space (can be refined based on dataset)
    cs_vals = [0.8, 0.97, 1.33]
    mt_vals = list(range(10, 121, 10))
    E_vals = [[10, 15, 20], [8, 12, 16], [7.5, 10, 14]]
    L_vals = [[10, 10, 10], [20, 5, 5], [5, 5, 20]]

    matched_configs = []

    for cs, mt, E, L in itertools.product(cs_vals, mt_vals, E_vals, L_vals):
        input_batch = {
            'cs': torch.tensor(cs).unsqueeze(0),
            'mt': torch.tensor(mt).unsqueeze(0),
            'E': [torch.tensor([e]) for e in E],
            'L': [torch.tensor([l]) for l in L],
        }
        with torch.no_grad():
            a_hat_pred = model(input_batch).item()
            if abs(a_hat_pred - target_a_hat) < tolerance:
                matched_configs.append((mt, E, L, cs, a_hat_pred))

    # Output matching configurations
    print(f"Target a_hat = {target_a_hat}, tolerance = {tolerance}")
    if not matched_configs:
        print("No matching configurations found.")
    else:
        print(f"Found {len(matched_configs)} matching configurations:")
        for config in matched_configs:
            mt, E, L, cs, pred = config
            print(f"Match: a_hat_pred={pred:.6f} -> mt={mt}, E={E}, L={L}, cs={cs}")
    return matched_configs
    

if __name__ == '__main__':
    reverse_search(target_a_hat=0.00235, tolerance=2e-5)
