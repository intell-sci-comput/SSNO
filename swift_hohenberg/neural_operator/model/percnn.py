import torch
import torch.nn as nn
import numpy as np
import math

lap_2d_op = [[[[0, 0, -1/12, 0, 0],
               [0, 0, 4/3, 0, 0],
               [-1/12, 4/3, -5, 4/3, -1/12],
               [0, 0, 4/3, 0, 0],
               [0, 0, -1/12, 0, 0]]]]

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'PeRCNN'
        self.hidden_channels = args.width
        self.mu_up = 1.0
        self.dx = 8 * math.pi / args.size
        self.dt = args.dt

        self.CA = nn.Parameter(torch.tensor([(np.random.rand() - 0.5) * 2], dtype=torch.float32))

        # ✅ 所有卷积统一保持尺寸：padding=2，且使用周期边界
        self.W_laplace = nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False, padding_mode='circular')
        self.W_laplace.weight.data = (1 / self.dx**2) * torch.tensor(lap_2d_op, dtype=torch.float32)
        self.W_laplace.weight.requires_grad = False

        self.Wh1_u = nn.Conv2d(1, self.hidden_channels, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')
        self.Wh2_u = nn.Conv2d(1, self.hidden_channels, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')
        self.Wh4_u = nn.Conv2d(self.hidden_channels, 1, kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')

        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh4_u]
        self.init_filter(self.filter_list, c=0.02)

    def init_filter(self, filter_list, c):
        for filt in filter_list:
            nn.init.xavier_uniform_(filt.weight)
            filt.weight.data = c * filt.weight.data
            if filt.bias is not None:
                filt.bias.data.fill_(0.0)

    def one_step(self, h):
        # NHWC -> NCHW
        h = h.permute(0, 3, 1, 2)            # [B,1,H,W]
        u_prev = h[:, 0:1, ...]              # [B,1,H,W]

        # ✅ 两路的空间尺寸现在一致（都为 [B,1,H,W]）
        lap_term = self.W_laplace(u_prev)
        nonlin = self.Wh4_u(self.Wh1_u(u_prev) * self.Wh2_u(u_prev))

        u_res = self.mu_up * torch.sigmoid(self.CA) * lap_term + nonlin
        u_next = u_prev + u_res * self.dt

        return u_next.permute(0, 2, 3, 1)    # 回到 NHWC

    def forward(self, u, n_step=1):
        traj = []
        for _ in range(n_step):
            u = u.squeeze(1)         # [B, H, W, 1]
            u = self.one_step(u)
            u = u.unsqueeze(1)       # [B, 1, H, W, 1]
            traj.append(u)
        return torch.cat(traj, dim=1)
