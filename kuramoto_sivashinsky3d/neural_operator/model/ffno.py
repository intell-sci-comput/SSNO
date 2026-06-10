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


class SpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, modes3, factor=4,
                 n_ff_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1 = modes1
        self.n_modes2 = modes2
        self.n_modes3 = modes3
        n_modes = [modes1, modes2, modes3]
  
        self.fourier_weight = nn.ParameterList([])
        for i in range(3):
            weight = torch.FloatTensor(in_dim, out_dim, n_modes[i], 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm)

    def forward(self, x):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x)
        return b

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n z i -> b i m n z')
        B, I, M, N, Z = x.shape

        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N, Z)
        out_ft[:, :, :self.n_modes1, :, :] = torch.einsum(
                "bixyz,iox->boxyz", x_ftx[:, :, :self.n_modes1, :, :],
                torch.view_as_complex(self.fourier_weight[0]))
        xx = torch.fft.irfft(out_ft, n=M, dim=-3, norm='ortho')

        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1, Z)
        out_ft[:, :, :, :self.n_modes2, :] = torch.einsum(
                "bixyz,ioy->boxyz", x_fty[:, :, :, :self.n_modes2, :],
                torch.view_as_complex(self.fourier_weight[1]))
        xy = torch.fft.irfft(out_ft, n=N, dim=-2, norm='ortho')
        
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, I, M, N, Z // 2 + 1)
        out_ft[:, :, :, :, :self.n_modes3] = torch.einsum(
                "bixyz,ioz->boxyz", x_ftz[:, :, :, :, :self.n_modes3],
                torch.view_as_complex(self.fourier_weight[2]))
        xz = torch.fft.irfft(out_ft, n=Z, dim=-1, norm='ortho')

        x = xx + xy + xz
        x = rearrange(x, 'b i m n z -> b m n z i')
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'FFNO'
        self.modes1 = args.modes
        self.modes2 = args.modes
        self.modes3 = args.modes
        self.width = args.width
        self.in_dim = args.dim + 3
        self.out_dim = args.dim 
        self.n_layers = args.n_layers

        self.in_proj = nn.Linear(self.in_dim, self.width)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv3d(self.width,
                                                       self.width,
                                                       self.modes1, self.modes2, self.modes3))
        self.out = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.out_dim))
        
        self.grid = self.get_grid(shape=(1, args.size, args.size, args.size))

    def one_step(self, x):
        batch_size = x.shape[0]
        grid = self.grid.to(x.device)
        x = torch.cat((x, grid.repeat(batch_size, 1, 1, 1, 1)), dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.in_proj(x.permute(0, 2, 3, 4, 1))
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
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_z, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
