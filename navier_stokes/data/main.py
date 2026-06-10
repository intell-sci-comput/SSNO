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
import os
import gc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of data generation')

    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Used device')
        
    parser.add_argument('--path', type=str, default='/home/zhangrui/ZEV_C/SSNO/navier_stokes',
                help='path to save the data')

    parser.add_argument('--mode', type=str, default='train',
            help='train or test')
    
    parser.add_argument('--s', type=int, default=512,
            help='data original size')
    
    parser.add_argument('--N', type=int, default=5,
            help='Number of the data generation')
    
    parser.add_argument('--re', type=float, default=1000,
            help='1/Re in NSE')
    
    parser.add_argument('--T', type=float, default=10.0,
            help='final time')

    parser.add_argument('--dt', type=float, default=1e-4,
            help='dt')

    parser.add_argument('--record_ratio', type=int, default=100,
            help='record ratio')
    


    cfg = parser.parse_args()
    for re in [2000]:
        for mode in ['test', 'val', 'train']:
            cfg.mode = mode
            cfg.re = re
            if mode == 'train':
                cfg.N = 5
                cfg.T = 8
            if mode == 'val':
                cfg.N = 2
                cfg.T = 8
            if mode == 'test':
                cfg.N = 5
                cfg.T = 15
            generate_ns_data(cfg)


            gc.collect()
            torch.cuda.empty_cache()
            cfg = parser.parse_args()  # 重新parse，避免被上面del掉
