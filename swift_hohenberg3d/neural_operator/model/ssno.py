"""
3D adaptation of the SSNO model updated to use real FFT (rfftn / irfftn).
Key changes:
- Use torch.fft.rfftn / torch.fft.irfftn for efficiency on real-valued problems.
- Spectral axis (last axis) now has size Wc = W//2 + 1 in spectral domain; adjust shapes accordingly.
- Build a spectral-compatible dealias mask (dealias_spectral) of shape (1, D, H, Wc) and kx vector in [0..Wc-1].
- Keep complex operations in spectral domain; linear layers that previously used complex dtype are preserved.
- Updated helper transforms fftn_physical / ifftn_spectral to rfftn/irfftn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _get_act(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation {name}")


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = "SSNO"

        self.dim = args.dim
        self.channel = args.channel
        self.k_num = args.k_num
        self.dt = args.dt
        self.anti_alias_ratio = args.anti_alias_ratio
        self.width = args.width
        self.act = _get_act(args.act)
        self.k_max = getattr(args, "k_max", None)

        self.use_fu = args.use_fu
        self.use_fx = args.use_fx
        self.use_pi = args.use_pi

        self.linop_mlp = nn.Sequential(
            nn.Linear(4, self.width), self.act,
            nn.Linear(self.width, self.width), self.act,
            nn.Linear(self.width, 2),
        )

        if self.use_pi == 1:
            self.k_layer = nn.Sequential(
                nn.Linear(4, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.k_num * 2)
            )

            # still operate in complex spectral domain for learned projections
            self.l1 = nn.Linear((self.k_num + 1) * self.dim, self.channel, dtype=torch.complex64)
            self.l2 = nn.Linear((self.k_num + 1) * self.dim, self.channel, dtype=torch.complex64)
            self.mask_param = nn.Parameter(torch.ones(self.channel))
            self.conv11 = nn.Conv3d(self.channel, self.dim, 1, 1)

        if self.use_fx==1:
            self.f_x = nn.Sequential(
                nn.Linear(3, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.dim),
            )

        if self.use_fu ==1:
            self.f_u = nn.Sequential(
                nn.Linear(self.dim, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.dim),
            )

        self._cache = {}

    @staticmethod
    def compute_features(x_h, xg):
        # x_h: (B, C, D, H, Wc) complex spectral coefficients from rfftn
        # xg: (k_num, D, H, Wc) complex
        B, C, D, H, Wc = x_h.shape
        n = xg.shape[0]
        xg_expanded = xg.unsqueeze(0).unsqueeze(0)            # (1,1,k_num,D,H,Wc)
        x_h_expanded = x_h.unsqueeze(2)                       # (B,C,1,D,H,Wc)
        product = (x_h_expanded * xg_expanded).contiguous().view(B, C * n, D, H, Wc)
        return torch.cat([x_h, product], dim=1)

    def _build_freq_cache(self, D, H, W, device):
        k_max = min(D, H, W) // 2
        Wc = W//2 + 1
        key = (H, W, device, self.k_max)
        kz = torch.fft.fftfreq(D, d=1.0, device=device).reshape(D, 1, 1).repeat(1, H, Wc)
        ky = torch.fft.fftfreq(H, d=1.0, device=device).reshape(1, H, 1).repeat(D, 1, Wc)
        kx = torch.fft.rfftfreq(W, d=1.0, device=device).reshape(1, 1, Wc).repeat(D, H, 1)

        # 广播到同一形状 (D, H, W//2+1)
        mask = (
            (torch.abs(kz) <= self.anti_alias_ratio * k_max)
            & (torch.abs(ky) <= self.anti_alias_ratio * k_max)
            & (torch.abs(kx) <= self.anti_alias_ratio * k_max)
        )

        dealias = mask.float().unsqueeze(0)   # (1, D, H, Wc)

        t_d = torch.linspace(0, 1, D, device=device)
        t_h = torch.linspace(0, 1, H, device=device)
        t_w = torch.linspace(0, 1, W, device=device)
        Z, Y, X = torch.meshgrid(t_d, t_h, t_w, indexing="ij")
        XYZ = torch.stack([Z, Y, X], dim=-1)

        k_norm = (kx**2 + ky**2 + kz**2).sqrt()

        cache = {"kx": kx, "ky": ky, "kz": kz, "k_norm": k_norm, "dealias": dealias, "XYZ": XYZ, "k_max": k_max, "Wc": Wc}
        self._cache[key] = cache
        return cache

    def nonlinear_only(self, u, fx, xg, dealias):
        # u: (B, D, H, W, C) real-valued physical field
        x = u.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        B, C, D, H, Wfull = x.shape

        r = torch.zeros(x.shape, device=x.device)
        if self.use_pi == 1:
            # spectral transform: real->complex, with reduced last axis (Wc)
            x_h = torch.fft.rfftn(x, dim=(2, 3, 4))   # (B, C, D, H, Wc) complex

            xp = self.compute_features(x_h, xg)      # (B, C*(k_num+1), D, H, Wc)
            feat = xp.permute(0, 2, 3, 4, 1)          # (B, D, H, Wc, feat)

            x_tp1 = self.l1(feat).permute(0, 4, 1, 2, 3)  # (B, channel, D, H, Wc)
            x_tp2 = self.l2(feat).permute(0, 4, 1, 2, 3)

            # bring to physical (real) domain for pointwise nonlinear prod
            # irfftn reconstructs to full W
            g1 = torch.fft.irfftn(x_tp1, s=(D, H, Wfull), dim=(2, 3, 4))
            g2 = torch.fft.irfftn(x_tp2, s=(D, H, Wfull), dim=(2, 3, 4))

            prod = g1 * g2 * self.mask_param.view(1, -1, 1, 1, 1)

            # go back to spectral (rfftn) and apply spectral dealias mask (spectral shape)
            nl_h = torch.fft.rfftn(prod, dim=(2, 3, 4)) * dealias.unsqueeze(1)

            phys = torch.fft.irfftn(nl_h, s=(D, H, Wfull), dim=(2, 3, 4))
            r = r + self.conv11(phys)

        if self.use_fu == 1:
            # f_u expects last dim = channels (real physical), returns physical real
            r = r + self.f_u(u).permute(0, 4, 1, 2, 3)

        if self.use_fx == 1:
            r = r + fx.permute(0, 4, 1, 2, 3)
        return r.permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, u, n_step=1, dt=None):
        if dt is None:
            dt = self.dt
        u = u.squeeze(1)
        B, D, H, W, C = u.shape
        device = u.device

        consts = self._build_freq_cache(D, H, W, device)
        kx, ky, kz, k_norm, dealias, XYZ = consts["kx"], consts["ky"], consts["kz"], consts["k_norm"], consts["dealias"], consts["XYZ"]
        Wc = consts["Wc"]

        if self.use_fx == 1:
            fx = self.f_x(XYZ.reshape(-1, 3)).reshape(D, H, W, 2 * self.dim).unsqueeze(0)
        else:
            fx = torch.zeros(B, D, H, W, 2 * self.dim, device=device)

        if self.use_pi == 1:
            xg_realimag = self.k_layer(torch.stack([kx, ky, kz, k_norm], dim=-1))
            k_num = xg_realimag.shape[-1] // 2
            # xg_realimag already expected to be shaped (D, H, Wc, k_num, 2)
            xg_realimag = xg_realimag.view(D, H, Wc, k_num, 2)
            xg = xg_realimag[..., 0] + 1j * xg_realimag[..., 1]
            xg = xg.permute(3, 0, 1, 2).contiguous()   # (k_num, D, H, Wc)
        else:
            xg = None

        ab = self.linop_mlp(torch.stack([kx, ky, kz, k_norm], dim=-1))
        a, b = ab[..., 0], ab[..., 1]
        lam = -F.softplus(a) + 1j * b
        E = torch.exp(lam * dt).unsqueeze(0).unsqueeze(0)
        E2 = torch.exp(lam * (dt / 2.0)).unsqueeze(0).unsqueeze(0)

        def N(u_real):
            return self.nonlinear_only(u_real, fx=fx, xg=xg, dealias=dealias)

        def fftn_physical(x):
            # real->reduced complex spectral (rfftn)
            return torch.fft.rfftn(x.permute(0, 4, 1, 2, 3), dim=(2, 3, 4))

        def ifftn_spectral(X):
            # reduced complex -> full real physical (irfftn)
            return torch.fft.irfftn(X, s=(D, H, W), dim=(2, 3, 4)).permute(0, 2, 3, 4, 1).contiguous()

        U = fftn_physical(u)
        traj = []

        for _ in range(n_step):
            u_real = ifftn_spectral(U)
            a = dt * N(u_real)
            arg_b = ifftn_spectral(E2 * fftn_physical(u_real + a / 2.0))
            b_stage = dt * N(arg_b)
            u_e2 = ifftn_spectral(E2 * fftn_physical(u_real))
            arg_c = u_e2 + b_stage / 2.0
            c_stage = dt * N(arg_c)
            u_e = ifftn_spectral(E * fftn_physical(u_real))
            c_e2 = ifftn_spectral(E2 * fftn_physical(c_stage))
            arg_d = u_e + c_e2
            d = dt * N(arg_d)
            a_e = ifftn_spectral(E * fftn_physical(a))
            bc_e2 = ifftn_spectral(E2 * fftn_physical(b_stage + c_stage))
            u_next = u_e + (1.0 / 6.0) * (a_e + 2.0 * bc_e2 + d)
            U = fftn_physical(u_next)
            traj.append(u_next)

        return torch.stack(traj, dim=1)
