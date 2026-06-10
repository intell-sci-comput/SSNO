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
            nn.Linear(3, self.width), self.act,
            nn.Linear(self.width, self.width), self.act,
            nn.Linear(self.width, 2),
        )

        if self.use_pi==1:
            self.k_layer = nn.Sequential(
                nn.Linear(3, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.k_num * 2)
            )

            self.l1 = nn.Linear((self.k_num + 1) * self.dim, self.channel, dtype=torch.complex64)
            self.l2 = nn.Linear((self.k_num + 1) * self.dim, self.channel, dtype=torch.complex64)
            self.mask_param = nn.Parameter(torch.ones(self.channel))
            self.conv11 = nn.Conv2d(self.channel, self.dim, 1, 1)
        
        if self.use_fx==1:
            self.f_x = nn.Sequential(
                nn.Linear(2, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.dim),
            )

        if self.use_fu==1:
            self.f_u = nn.Sequential(
                nn.Linear(self.dim, self.width), self.act,
                nn.Linear(self.width, self.width), self.act,
                nn.Linear(self.width, self.dim),
            )

        self._cache = {}

    @staticmethod
    def compute_features(x_h, xg):
        b, c, h, w = x_h.shape
        n = xg.shape[0]
        xg_expanded = xg.unsqueeze(0).unsqueeze(1)
        x_h_expanded = x_h.unsqueeze(2)
        product = (x_h_expanded * xg_expanded).contiguous().view(b, c * n, h, w)
        return torch.cat([x_h, product], dim=1)

    def _build_freq_cache(self, H, W, device):
        key = (H, W, device, self.k_max)
        if key in self._cache: return self._cache[key]
        Wr = W // 2 + 1
        k_max = math.floor(H / 2.0) if self.k_max is None else int(self.k_max)
        ky_full = torch.cat((torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0).float()
        ky = ky_full.repeat(H, 1)
        kx = ky.t().contiguous()[..., :Wr]
        ky = ky[..., :Wr]
        mask = (torch.abs(ky) <= self.anti_alias_ratio * k_max) & (torch.abs(kx) <= self.anti_alias_ratio * k_max)
        dealias = mask.float().unsqueeze(0)
        t = torch.linspace(0, 1, H + 1, device=device)[:-1]
        X, Y = torch.meshgrid(t, t, indexing="ij")
        XY = torch.stack([X, Y], dim=-1)
        k_norm = (kx**2 + ky**2).sqrt()
        cache = {"kx": kx, "ky": ky, "k_norm": k_norm, "dealias": dealias, "XY": XY, "k_max": k_max, "Wr": Wr}
        self._cache[key] = cache
        return cache

    def nonlinear_only(self, u, fx, xg, dealias):
        x = u.permute(0, 3, 1, 2)
        B, C, H, W = x.shape

        nl_h = torch.zeros(B, self.channel, H, W // 2 + 1, device=x.device, dtype=torch.complex64)
        if self.use_pi==1:
            x_h = torch.fft.rfft2(x)
            xp = self.compute_features(x_h, xg)
            x_tp1 = self.l1(xp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_tp2 = self.l2(xp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            g1 = torch.fft.irfft2(x_tp1, s=(H, W)).real
            g2 = torch.fft.irfft2(x_tp2, s=(H, W)).real
            prod = g1 * g2 * self.mask_param.view(1, -1, 1, 1)
            nl_h = torch.fft.rfft2(prod) * dealias.unsqueeze(1)


        r = self.conv11(torch.fft.irfft2(nl_h, s=(H, W)))

        if self.use_fu==1:
            r = r + self.f_u(u).permute(0, 3, 1, 2)

        if self.use_fx==1:
            r = r + fx
        return r.permute(0, 2, 3, 1).contiguous()

    def forward(self, u, n_step=1, dt=None):
        if dt is None: dt = self.dt
        u = u.squeeze(1)
        B, H, W, C = u.shape
        device = u.device

        consts = self._build_freq_cache(H, W, device)
        kx, ky, k_norm, dealias, XY = consts["kx"], consts["ky"], consts["k_norm"], consts["dealias"], consts["XY"]

        if self.use_fx==1:
            fx = self.f_x(XY).permute(2, 0, 1).unsqueeze(0).to(u.dtype)
        else:
            fx = torch.zeros(B, self.dim, H, W, device=device, dtype=u.dtype)

        if self.use_pi==1:
            xg_realimag = self.k_layer(torch.stack([kx, ky, k_norm], dim=-1))
            k_num = xg_realimag.shape[-1] // 2
            xg_realimag = xg_realimag.view(H, W // 2 + 1, k_num, 2)
            xg = xg_realimag[..., 0] + 1j * xg_realimag[..., 1]
            xg = xg.permute(2, 0, 1).contiguous()
        else:
            xg = None

        ab = self.linop_mlp(torch.stack([kx, ky, k_norm], dim=-1))
        a, b = ab[..., 0], ab[..., 1]
        lam = -F.softplus(a) + 1j * b
        E = torch.exp(lam * dt).unsqueeze(0).unsqueeze(0)
        E2 = torch.exp(lam * (dt / 2.0)).unsqueeze(0).unsqueeze(0)

        def N(u_real):
            return self.nonlinear_only(u_real, fx=fx, xg=xg, dealias=dealias)

        def rfft2_physical(x): return torch.fft.rfft2(x.permute(0, 3, 1, 2))
        def irfft2_spectral(X): return torch.fft.irfft2(X, s=(H, W)).permute(0, 2, 3, 1).contiguous()

        U = rfft2_physical(u)
        traj = []

        for _ in range(n_step):
            u_real = irfft2_spectral(U)
            a = dt * N(u_real)
            arg_b = irfft2_spectral(E2 * rfft2_physical(u_real + a / 2.0))
            b_stage = dt * N(arg_b)
            u_e2 = irfft2_spectral(E2 * rfft2_physical(u_real))
            arg_c = u_e2 + b_stage / 2.0
            c_stage = dt * N(arg_c)
            u_e = irfft2_spectral(E * rfft2_physical(u_real))
            c_e2 = irfft2_spectral(E2 * rfft2_physical(c_stage))
            arg_d = u_e + c_e2
            d = dt * N(arg_d)
            a_e = irfft2_spectral(E * rfft2_physical(a))
            bc_e2 = irfft2_spectral(E2 * rfft2_physical(b_stage + c_stage))
            u_next = u_e + (1.0 / 6.0) * (a_e + 2.0 * bc_e2 + d)
            U = rfft2_physical(u_next)
            traj.append(u_next)

        return torch.stack(traj, dim=1)

