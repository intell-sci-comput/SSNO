import torch
import numpy as np
import random
import math
import json
import sys
import os
import time
from nse import navier_stokes_2d, GaussianRF
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
    re = cfg.re
    nu = 1.0 / cfg.re
    mode = cfg.mode
    path = cfg.path

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
    with open(f'{log_save_path}/ns_{mode}_re_{re}.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)
    sys.stdout.flush()
    
    GRF = GaussianRF(s, device=device)
    
    t = 2 * math.pi * torch.linspace(0, 1, s+1, device=device)
    t = t[0: -1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * torch.cos(8*X).to(device)

    bsize = min(100, N)
    c = 0
    u = torch.zeros(N, record_steps+1, s, s, 1)
    

    for j in range(N//bsize):
        # Sample random feilds
        w0 = GRF(bsize)
        visc = nu * torch.ones(bsize, device=device)
        sol, sol_t = navier_stokes_2d(w0, f, visc, T, dt, record_steps)
        sol = torch.concat([w0[..., None].to('cpu'), sol.to('cpu')], dim=3)
        u[c:(c+bsize),...] = sol.unsqueeze(-1).permute(0, 3, 1, 2, 4)
        c += bsize
        print(j, c)
        print(u.max())
    torch.save(u, f'{data_save_path}/ns_{mode}_re_{re}')
