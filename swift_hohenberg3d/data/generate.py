import torch
import numpy as np
import random
import math
import json
import sys
import os
import time
from she import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = True


def generate_ns_data(cfg):
    device = cfg.device
    s = cfg.s
    dt = cfg.dt
    T = cfg.T
    record_steps = int(cfg.record_ratio * T)
    N = cfg.N
    mode = cfg.mode
    path = cfg.path
    sup = cfg.sup

    if mode == 'train':
        setup_seed(0)
    if mode == 'test':
        setup_seed(1)
    if mode == 'val':
        setup_seed(2)
        
    data_save_path = f'{path}/dataset'
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    
    log_save_path = f'{path}/log'
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    with open(f'{log_save_path}/sh3d_{mode}.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)
    sys.stdout.flush()
    
    GRF = GaussianRF(s * sup, device=device)
    
    bsize = min(100, N)
    c = 0
    u = torch.zeros(N, record_steps+1, s, s, s, 1, device=device) 
    

    for j in range(N//bsize):
        # Sample random feilds
        u0 = GRF(bsize)
        sol = sh_3d_rk4(u0=u0, T=T, dt=dt, record_steps=record_steps)
        u[c:(c+bsize),...] = sol.unsqueeze(-1).permute(0, 4, 1, 2, 3, 5)[:, :, ::sup, ::sup, ::sup]
        c += bsize
        print(j, c)
    torch.save(u, f'{data_save_path}/sh3d_{mode}.pt')
