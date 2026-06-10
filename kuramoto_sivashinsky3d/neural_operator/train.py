import sys
import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from tools import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

EPS = 1e-7

def test(net, test_data, final_t, print_step=15):
    net.eval()
    num = test_data.shape[0]
    n_step = test_data.shape[1] - 1

    start_time = time.time()
    with torch.no_grad():
        pre_data = net(test_data[:, 0:1], n_step=n_step)
    inference_time = time.time() - start_time

    # Sampling time steps for reporting
    sub_t = max(1, n_step // print_step)
    gth = test_data[:, 1:][:, ::sub_t]  # [B, T_sub, ...]
    pre = pre_data[:, ::sub_t]

    error_list = []
    corr_list = []
    corr_per_sample = []  # <-- store per-sample correlation curves

    for i in range(gth.shape[1]):
        error = torch.norm((pre[:, i] - gth[:, i]).reshape(num, -1), dim=1) \
                / torch.norm((gth[:, i]).reshape(num, -1), dim=1)
        error_val = error.mean().item()
        error_list.append(error_val)

        x = torch.cat([pre[:, i].reshape(num, -1),
                    pre[:, i].reshape(num, -1)], dim=1)
        y = torch.cat([gth[:, i].reshape(num, -1),
                    gth[:, i].reshape(num, -1)], dim=1)
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        corr = (vx * vy).sum(dim=1) / (
            torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy**2).sum(dim=1)) + 1e-8
        )  # shape: [B]

        corr_val = corr.mean().item()
        corr_list.append(corr_val)
        corr_per_sample.append(corr.cpu().numpy())  # save per-sample

        print(f"Step {i:3d} | Error: {error_val:.6f} | Corr: {corr_val:.6f}")

    # Convert to [B, T] array
    corr_per_sample = np.stack(corr_per_sample, axis=1)

    # Compute HCT per sample then average
    hct_09_list = []
    hct_08_list = []
    for b in range(num):
        below09 = np.where(corr_per_sample[b] < 0.9)[0]
        below08 = np.where(corr_per_sample[b] < 0.8)[0]
        hct_09_list.append(below09[0] if below09.size > 0 else print_step)
        hct_08_list.append(below08[0] if below08.size > 0 else print_step)

    hct_09 = float(np.mean(hct_09_list)) / print_step * final_t
    hct_08 = float(np.mean(hct_08_list)) / print_step * final_t

    mean_error = np.mean(error_list)
    mean_corr = np.mean(corr_list)

    print("Mean Error:", mean_error)
    print("Mean Corr :", mean_corr)
    print("Avg First corr < 0.90:", hct_09)
    print("Avg First corr < 0.80:", hct_08)

    return mean_error, mean_corr, hct_09, hct_08, inference_time




def train(args, net):
    num_parameter = param_flops(net)
    print(f"Number of parameters: {num_parameter}")
    model = args.model
    model_name = args.model_name
    device = args.device
    data_path = args.data_path
    print_step = args.print_step
    size = args.size
    dim = args.dim

    data_dict_path = os.path.join(data_path, f'log/gl_train.txt')
    with open(data_dict_path, 'r') as file:
        data_dict = json.load(file)
    data_ratio = data_dict['record_ratio']
    data_size = data_dict['s']
    final_t = data_dict['T']
    sub_x = data_size // size

    dt = args.dt
    model_ratio = round(1 / dt)
    num_train = args.num_train
    batch_size = args.batch_size

    # data.shape = [N, T, s, s, dim]
    train_data = torch.load(os.path.join(data_path, f'dataset/gl_train.pt'), map_location='cpu')[:num_train,
                 ::int(data_ratio // model_ratio), ::sub_x, ::sub_x, ::sub_x, :].to(device)
    test_data = torch.load(os.path.join(data_path, f'dataset/gl_val.pt'), map_location='cpu')[...,
                 ::int(data_ratio // model_ratio), ::sub_x, ::sub_x, ::sub_x, :].to(device)
    val_data = torch.load(os.path.join(data_path, f'dataset/gl_val.pt'), map_location='cpu')[...,
                 ::int(data_ratio // model_ratio), ::sub_x, ::sub_x, ::sub_x, :].to(device)

    size, train_step = train_data.shape[2], train_data.shape[1]
    sparse_coeff = args.sparse_coeff
    num_step1, num_step2 = args.num_step1, args.num_step2

    save_dir = os.path.join(data_path, f"log_model/experiment_results_model_{model}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(data_path, "log_train_history"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "log_results"), exist_ok=True)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.num_iterations + 1, max_lr=args.lr)

    train_loss_list, val_error_list = [], []
    train_time, infer_time = [], []
    best_val_error = 1e10

    for step in range(args.num_iterations + 1):
        net.train()
        start_time = time.time()
        input_data = torch.zeros(batch_size, 1, size, size, size, dim, device=device) 
        output_data = torch.zeros(batch_size, num_step2, size, size, size, dim, device=device) 

        n = random.randint(0, num_step1)
        max_start_t = train_step - n - num_step2 - 1
        possible_start_t = list(range(0, max_start_t + 1))
        random_indices = random.sample(range(num_train), batch_size)
        sampled_start_t = random.choices(possible_start_t, k=batch_size)

        for k_batch in range(batch_size):
            traj_idx = random_indices[k_batch]
            start_t = sampled_start_t[k_batch]
            input_data[k_batch] = train_data[traj_idx, start_t:start_t + 1]
            output_data[k_batch] = train_data[traj_idx, start_t + n + 1:start_t + n + 1 + num_step2]

        if n > 0:
            with torch.no_grad():
                input_data = net(input_data, n_step=n)[:, -1:]

        pre_data = net(input_data, n_step=num_step2)
        loss = torch.mean(torch.abs(pre_data - output_data) ** 2)
        if sparse_coeff > 0:
            loss = loss + sparse_coeff * torch.sum(torch.abs(net.mask_param))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        train_time.append(time.time() - start_time)
        train_loss_list.append(loss.item())

        if step % 100 == 0:
            print(f"\nTrain epoch {step} | Training loss: {loss.item():.6f}")
            print("Validation ----------------")
            val_error, _, _, _, val_time = test(net, val_data, final_t, print_step=print_step)
            if sparse_coeff > 0:
                print(f"Mask sum: {torch.sum(torch.abs(net.mask_param) > 1e-4).item():.6f}")
            val_error_list.append(val_error)
            infer_time.append(val_time)

            if val_error < best_val_error:
                best_val_error = val_error
                print("Test ----------------")
                test(net, test_data, final_t, print_step=print_step)
                torch.save({
                    'model_state': net.state_dict(),
                }, os.path.join(save_dir, f'{model_name}.pt'))
                print("----------- SAVING NEW MODEL -----------")

            sys.stdout.flush()

    print("\n---------------------------- FINAL RESULT -----------------------------")
    ckpt = torch.load(os.path.join(save_dir, f'{model_name}.pt'))
    net.load_state_dict(ckpt['model_state'])
    net.to(device)

    print("Validation ----------------")
    final_val_error, val_mean_corr, val_hct_09, val_hct_08, _ = test(net, val_data, final_t, print_step=print_step)
    print("Test ----------------")
    final_test_error, test_mean_corr, test_hct_09, test_hct_08, _ = test(net, test_data, final_t, print_step=print_step)

    np.savez_compressed(os.path.join(data_path, "log_train_history", f"{model_name}_history.npz"),
                        train_loss=np.array(train_loss_list),
                        val_error=np.array(val_error_list))

    train_time = torch.median(torch.tensor(train_time)).item()
    infer_time = torch.median(torch.tensor(infer_time)).item()
    print(f"\nTraining time per epoch: {train_time:.8f} seconds")
    print(f"Average inference time: {infer_time:.8f} seconds")

    csv_path = os.path.join(data_path, "log_results",
                            f"experiment_results_size_{size}_dt_{dt}_model_{model}.csv")
    args_dict = vars(args)
    headers = list(args_dict.keys()) + ["num_params", "train_time", "infer_time", 
                                        "val_error", "val_mean_corr", "val_hct_09", "val_hct_08", 
                                        "test_error", "test_mean_corr", "test_hct_09", "test_hct_08"]
    row = [args_dict[h] for h in args_dict] + [num_parameter, train_time, infer_time,
                                               final_val_error, val_mean_corr, val_hct_09, val_hct_08, 
                                               final_test_error, test_mean_corr, test_hct_09, test_hct_08]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
