
import torch
import matplotlib.pyplot as plt
import pandas as pd
from model.MLP_Att import MLPWithAttention

CS_OPTIONS = [0.8, 0.97, 1.33]
E_OPTIONS = [[10, 15, 20], [8, 12, 16], [7.5, 10, 14]]
L_OPTIONS = [[10, 10, 10], [20, 5, 5], [5, 5, 20]]

CS_RANGE = (0.5, 2.0)
E_RANGE = (5.0, 50.0)
L_RANGE = (0.0, 30.0)

PLOT_DIR = "Inverse_Results/Plot"
CSV_DIR = "Inverse_Results/csv"

def clamp_inputs(cs, E, L):
    cs.data.clamp_(*CS_RANGE)
    E.data.clamp_(*E_RANGE)
    L.data.clamp_(*L_RANGE)

def round_to_training_values(val, options):
    for opt in options:
        if abs(val - opt) / opt <= 0.05: # Check if within 5% of the option
            return opt
    return val

def round_list_to_training_values(lst, options_list):
    for opts in options_list:
        if all(abs(x - y) / y <= 0.05 for x, y in zip(lst, opts)): # Check if all elements are within 5% of the corresponding option
            return opts
    return lst

def inverse_optimize_single_fixed_mt(a_target, mt_fixed, max_iter=3000, lr=0.01, seeds=[42, 123, 888]):
    best_result = None
    min_loss = float('inf')
    all_seed_results = []

    for seed in seeds:
        torch.manual_seed(seed)

        cs = torch.tensor([1.0], requires_grad=True)
        E = torch.tensor([[15.0, 10.0, 7.5]], requires_grad=True)
        L = torch.tensor([[10.0, 10.0, 10.0]], requires_grad=True)
        mt = torch.tensor([mt_fixed])  # fixed

        optimizer = torch.optim.Adam([cs, E, L], lr=lr)

        input_dim, hidden_dim, output_dim = 10, 20, 1
        model = MLPWithAttention(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load("results/model.pt"))
        model.eval()
    # the model is now loaded and set to evaluation mode
        for _ in range(max_iter):
            optimizer.zero_grad()
            input_batch = {
                'cs': cs,
                'mt': mt,
                'E': [E[:, 0], E[:, 1], E[:, 2]],
                'L': [L[:, 0], L[:, 1], L[:, 2]]
            }
            pred = model(input_batch)
            loss = torch.abs(pred - a_target) # L1 loss
            loss.backward() # compute the gradients
            optimizer.step() # update the parameters
            clamp_inputs(cs, E, L) #clamp the values to their respective ranges

        final_loss = torch.abs(pred - a_target).item()

        cs_rounded = round_to_training_values(cs.item(), CS_OPTIONS)
        E_rounded = round_list_to_training_values(E.detach().numpy().flatten().tolist(), E_OPTIONS)
        L_rounded = round_list_to_training_values(L.detach().numpy().flatten().tolist(), L_OPTIONS)

        # Update the cs, E, L tensors with rounded values
        result = {
            'a_target': a_target,
            'a_pred': pred.item(),
            'mt': mt_fixed,
            'cs': cs_rounded,
            'E1': E_rounded[0],
            'E2': E_rounded[1],
            'E3': E_rounded[2],
            'L1': L_rounded[0],
            'L2': L_rounded[1],
            'L3': L_rounded[2],
            'seed': seed,
            'loss': final_loss
        }

        all_seed_results.append(result)

        if final_loss < min_loss:
            min_loss = final_loss
            best_result = result

    return best_result, all_seed_results

def batch_inverse_optimize_varying_mt(a_start=0.002, a_end=0.003, a_step=0.0005, mt_vals=range(10, 121, 10), **kwargs):
    a_vals = torch.arange(a_start, a_end + a_step, a_step)
    global_results = []
    all_seed_results = []

    for mt in mt_vals:
        print(f"Fixing mt = {mt}")
        for a in a_vals:
            print(f"Processing a_hat = {a:.6f}")
            best_result, all_results = inverse_optimize_single_fixed_mt(a.item(), mt_fixed=mt, **kwargs)
            global_results.append(best_result)
            all_seed_results.extend(all_results)
        print(f"mt = {mt} completed.")
    df_global = pd.DataFrame(global_results)
    #df_all = pd.DataFrame(all_seed_results)
    df_global['a_group'] = df_global['a_target'].round(4)
    
    df_global.to_csv(f"{CSV_DIR}/inverse_varying_mt_global.csv", index=False)
    #df_all.to_csv(f"{CSV_DIR}/inverse_varying_mt_all_seeds.csv", index=False)
    print("\nGlobal minima with varying mt saved to inverse_varying_mt_global.csv")
    #print("All seed results saved to inverse_varying_mt_all_seeds.csv")

    plt.figure(figsize=(18, 6))
    #We are now going to plot the results
    # Plotting the results for each a_hat
    # cs vs mt
    plt.subplot(1, 3, 1)
    for a in sorted(df_global['a_group'].unique()):
        sub = df_global[df_global['a_group'] == a]
        plt.plot(sub['mt'], sub['cs'], marker='o', label=f'a_hat={a}')
    plt.xlabel("mt")
    plt.ylabel("cs")
    plt.title("cs vs mt (by a_hat)")
    plt.grid(True)
    plt.legend()
    # E values vs mt
    plt.subplot(1, 3, 2)
    for a in sorted(df_global['a_group'].unique()):
        sub = df_global[df_global['a_group'] == a]
        plt.plot(sub['mt'], sub['E1'], marker='o', label=f'a_hat={a} E1')
        plt.plot(sub['mt'], sub['E2'], marker='o', label=f'a_hat={a} E2')
        plt.plot(sub['mt'], sub['E3'], marker='o', label=f'a_hat={a} E3')
    plt.xlabel("mt")
    plt.ylabel("E values")
    plt.title("E values vs mt (by a_hat)")
    plt.legend()
    plt.grid(True)
    # L values vs mt
    plt.subplot(1, 3, 3)
    for a in sorted(df_global['a_group'].unique()):
        sub = df_global[df_global['a_group'] == a]
        plt.plot(sub['mt'], sub['L1'], marker='o', label=f'a_hat={a} L1')
        plt.plot(sub['mt'], sub['L2'], marker='o', label=f'a_hat={a} L2')
        plt.plot(sub['mt'], sub['L3'], marker='o', label=f'a_hat={a} L3')
    plt.xlabel("mt")
    plt.ylabel("L values")
    plt.title("L values vs mt (by a_hat)")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Inverse Optimization Results with Varying mt")
    plt.subplots_adjust(top=0.85)
    # tight layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{PLOT_DIR}/inverse_varying_mt_combined_ahat_plot.png")
    print("Plot saved to inverse_varying_mt_combined_ahat_plot.png")

    return df_global

if __name__ == "__main__":
    batch_inverse_optimize_varying_mt()
