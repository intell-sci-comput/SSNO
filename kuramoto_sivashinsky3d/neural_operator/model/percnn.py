import torch
import torch.nn as nn
import math
import numpy as np

# --- laplace stencil (unchanged) ---
laplace_3d = np.zeros((1, 1, 5, 5, 5))
elements = [
    (-15/2, (0, 0, 0)),
    (4 / 3, (1, 0, 0)),
    (4 / 3, (0, 1, 0)),
    (4 / 3, (0, 0, 1)),
    (4 / 3, (-1, 0, 0)),
    (4 / 3, (0, -1, 0)),
    (4 / 3, (0, 0, -1)),
    (-1 / 12, (-2, 0, 0)),
    (-1 / 12, (0, -2, 0)),
    (-1 / 12, (0, 0, -2)),
    (-1 / 12, (2, 0, 0)),
    (-1 / 12, (0, 2, 0)),
    (-1 / 12, (0, 0, 2)),
]
for weight, (x, y, z) in elements:
    laplace_3d[0, 0, x+2, y+2, z+2] = weight


class upscaler(nn.Module):
    ''' Upscaler to convert low-res to high-res initial state '''
    def __init__(self):
        super(upscaler, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose3d(2, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        layers.append(nn.Sigmoid())
        layers.append(nn.ConvTranspose3d(8, 8, kernel_size=5, padding=5 // 2, stride=1, output_padding=0, bias=True))
        layers.append(nn.Conv3d(8, 2, 1, 1, padding=0, bias=True))
        self.convnet = nn.Sequential(*layers)

    def forward(self, h):
        return self.convnet(h)


class Model(nn.Module):
    ''' Recurrent convolutional neural network Cell (3D) '''
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'PeRCNN'

        # parameters
        self.input_channels  = args.dim     # e.g. 2
        self.hidden_channels = args.width
        self.dx = 6.28 / args.size
        self.dt = args.dt

        # trainable scalar
        self.CA = nn.Parameter(torch.tensor([(np.random.rand() - 0.5) * 2], dtype=torch.float32))

        # === Laplace conv: kernel=5, padding=2, circular, keep size ===
        self.W_laplace = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=2,
                                   bias=False, padding_mode='circular')
        self.W_laplace.weight.data = (1 / self.dx ** 2) * torch.tensor(laplace_3d, dtype=torch.float32)
        self.W_laplace.weight.requires_grad = False

        # === Nonlinear branch (all kernel=5, padding=2, circular) ===
        self.Wh1_u = nn.Conv3d(in_channels=self.input_channels, out_channels=self.hidden_channels,
                               kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')
        self.Wh2_u = nn.Conv3d(in_channels=self.input_channels, out_channels=self.hidden_channels,
                               kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')
        self.Wh3_u = nn.Conv3d(in_channels=self.input_channels, out_channels=self.hidden_channels,
                               kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')
        self.Wh4_u = nn.Conv3d(in_channels=self.hidden_channels, out_channels=1,
                               kernel_size=5, stride=1, padding=2, bias=True, padding_mode='circular')

        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u]
        self.init_filter(self.filter_list, c=0.01)

    def init_filter(self, filter_list, c):
        for filt in filter_list:
            nn.init.xavier_uniform_(filt.weight)
            filt.weight.data = c * filt.weight.data
            if filt.bias is not None:
                filt.bias.data.fill_(0.0)

    def one_step(self, h):
        # NHWDC -> NCDHW
        h = h.permute(0, 4, 1, 2, 3)        # [B, C, D, H, W]
        u_prev = h[:, 0:1, ...]             # first channel as u

        # All ops keep the same spatial size thanks to padding=2 + circular
        lap_term = self.W_laplace(u_prev)   # [B,1,D,H,W]
        nonlin   = self.Wh4_u(self.Wh1_u(h) * self.Wh2_u(h) * self.Wh3_u(h))  # [B,1,D,H,W]

        u_res  = 0.01 * torch.sigmoid(self.CA) * lap_term + nonlin
        u_next = u_prev + u_res * self.dt

        # put back to NHWDC with only u channel updated
        ch = u_next                          # if you have more fields, concat/stack here
        h  = ch.permute(0, 2, 3, 4, 1)       # [B, D, H, W, 1]
        return h

    def forward(self, u, n_step=1):
        traj = []
        for _ in range(n_step):
            u = u.squeeze(1)                 # [B, D, H, W, C]
            u = self.one_step(u)
            u = u.unsqueeze(1)               # [B, 1, D, H, W, C]
            traj.append(u)
        return torch.cat(traj, dim=1)
