import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
import os
from functools import reduce
import operator
from thop import profile
from dataclasses import dataclass
from model import model_dict
from typing import List, Tuple, Dict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_model(args):
    if args.model not in model_dict:
        raise ValueError(f"Unknown model: {args.model}")
    return model_dict[args.model](args)


def param_flops(net):
    params = 0
    for p in list(net.parameters()):
        params += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    print(' params: %.3f M' % (params / 1000000.0))
    return params


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
