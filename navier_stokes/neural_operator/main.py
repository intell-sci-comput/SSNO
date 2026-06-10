import math
import sys
import copy
import json
import csv
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

from train import train
from tools import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def main(args):
    now = datetime.now()
    timestring = f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    rand_suffix = f'{random.randint(0, 99999):05d}'
    timestring = f'{timestring}_{rand_suffix}'
    args.model_name = (
        f"{args.model}-seed-{args.seed}-{timestring}"
    )

    setup_seed(args.seed) 
    net = get_model(args).to(args.device)

    os.makedirs(f'{args.data_path}/log_train/log', exist_ok=True)
    os.makedirs(f'{args.data_path}/log_model', exist_ok=True)
    save_dir = os.path.join(args.data_path, f"log_model/experiment_results_re_{args.re}_model_{args.model}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "log_train_history"), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "log_results"), exist_ok=True)

    logfile = f'{args.data_path}/log_train/log/log_{args.model_name}.txt'
    sys.stdout = open(logfile, 'w')
    print('--------args----------')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('--------args----------\n')

    param_flops(net)
    sys.stdout.flush()
    train(args, net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters')

    # Basic settings
    parser.add_argument('--model', type=str, default="SSNO", help="Model name, must be in model_dict")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to use: cuda or cpu")
    parser.add_argument('--data_path', type=str, default="/home/zhangrui/zhangruiC/SSNO/navier_stokes",
                         help="Base data directory")
    parser.add_argument('--dim', type=int, default=1, help="Input/Output dimension (channels)")
    parser.add_argument('--size', type=int, default=128, help="Spatial resolution after downsampling")
    parser.add_argument('--re', type=int, default=500, help="re of nse")

    # Model hyperparameters
    parser.add_argument('--channel', type=int, default=32)
    parser.add_argument('--k_num', type=int, default=8)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--act', type=str, default='relu', help="Activation: relu / gelu / tanh")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step size")
    parser.add_argument('--use_pi', type=int, default=1, help="Whether to use the PI path")
    parser.add_argument('--use_fu', type=int, default=1, help="Whether to use the f_u path")
    parser.add_argument('--use_fx', type=int, default=1, help="Whether to use the f_x path")
    parser.add_argument('--anti_alias_ratio', type=float, default=0.67, help="Anti-aliasing ratio")
    parser.add_argument('--k_max', type=int, default=None, help="Optional max frequency cutoff (None = default H/2)")
    parser.add_argument('--n_layers', type=int, default=0, help='Number of layers for neural operator')
    parser.add_argument('--modes', type=int, default=0, help='Number of Fourier modes')
    parser.add_argument('--n_heads', type=int, default=0, help='Number of attention heads')
    
    # Training hyperparameters
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--num_train', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--print_step', type=int, default=15)

    # Time stepping parameters
    parser.add_argument('--num_step1', type=int, default=1, help="Max warmup steps before target horizon")
    parser.add_argument('--num_step2', type=int, default=4, help="Prediction steps per training iteration")

    # Sparse regularization
    parser.add_argument('--sparse_coeff', type=float, default=0.001)

    args = parser.parse_args()
    main(args)

