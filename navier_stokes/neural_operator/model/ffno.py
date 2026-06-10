import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, factor=4,
                 n_ff_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes

        self.fourier_weight = nn.ParameterList([])
        for _ in range(2):
            weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm)

    def forward(self, x):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x)
        return b

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, N, M // 2 + 1)

        out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, N // 2 + 1, M)
        out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'FFNO'
        self.modes = args.modes
        self.width = args.width
        self.in_dim = args.dim + 2
        self.out_dim = args.dim
        self.n_layers = args.n_layers

        self.in_proj = nn.Linear(self.in_dim, self.width)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=self.width,
                                                       out_dim=self.width,
                                                       n_modes=self.modes))
        self.out = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.out_dim))
        
        self.grid = self.get_grid(shape=(1, args.size, args.size, 1))
    
    def one_step(self, x):
        batch_size = x.shape[0]
        x = torch.cat((x, self.grid.to(x.device).repeat(batch_size, 1, 1, 1)), dim=-1)
        x = self.in_proj(x)
        x = F.gelu(x)
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b = layer(x)
            x = x + b    
        x = self.out(x)
        return x
    
    def forward(self, u, n_step=1):
        traj = []
        for _ in range(n_step):
            u = u.squeeze(1)
            u = self.one_step(u)
            u = u.unsqueeze(1)
            traj.append(u)
        return torch.cat(traj, dim=1)
    
    def get_grid(self, shape, device='cpu'):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
