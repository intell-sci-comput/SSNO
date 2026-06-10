import os
from generate import generate_ns_data

import json
import sys
import copy
from datetime import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of data generation')

    parser.add_argument('--device', type=str, default='cuda:5',
                        help='Used device')
        
    parser.add_argument('--path', type=str, default='/home/zhangrui/zhangruiC/SSNO/kuramoto_sivashinsky3dv2',
                help='path to save the data')

    parser.add_argument('--mode', type=str, default='train',
            help='train or test')
    
    parser.add_argument('--s', type=int, default=64,
            help='data saving size')

    parser.add_argument('--sup', type=int, default=2,
            help='sup*size: data original size')

    parser.add_argument('--N', type=int, default=5,
            help='Number of the data generation')
    
    parser.add_argument('--T', type=float, default=5.0,
            help='final time')

    parser.add_argument('--dt', type=float, default=4e-6,
            help='dt')

    parser.add_argument('--record_ratio', type=int, default=10,
            help='record ratio')
    
    cfg = parser.parse_args()
    for mode in ['train', 'val', 'test']:
        cfg.mode = mode
        if mode == 'train' or mode == 'val':
            cfg.N = 2
        if mode == 'test':
            cfg.N = 5
        generate_ns_data(cfg)
