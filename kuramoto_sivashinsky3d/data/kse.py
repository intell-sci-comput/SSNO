import torch
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GaussianRF(object):
    def __init__(self, size=128, dim=3, alpha=4, tau=8, sigma=None, boundary="periodic", device='cuda'):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = 0.5*tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

        k_x = wavenumers.transpose(1,2)
        k_y = wavenumers
        k_z = wavenumers.transpose(0,2)

        self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0)
        self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def __call__(self, N):
        coeff_r = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff_r = self.sqrt_eig * coeff_r
        u = torch.fft.ifftn(coeff_r, dim=list(range(-1, -self.dim - 1, -1))).real
        return u


def ks_3d_rk4(u0, T, dt=1e-4, record_steps=10):
    """
    3D Kuramoto–Sivashinsky solver with spectral RK4
    using real-to-complex FFT (rfft/irfft)
    u_t + Δ²u + Δu + 0.5*|∇u|² = 0
    """

    device = u0.device
    dtype = torch.float64 if u0.dtype == torch.float64 else torch.float32
    u0 = u0.to(dtype=dtype)

    batch = u0.shape[0]
    Nx, Ny, Nz = u0.shape[1], u0.shape[2], u0.shape[3]

    steps = math.ceil(T / dt)
    if record_steps <= 0:
        raise ValueError("record_steps must be >= 1")
    save_every = max(1, steps // record_steps)

    # frequency grids for rfft (last dim reduced)
    kx = 1/4 * torch.fft.fftfreq(Nx, d=1.0/Nx).to(device)
    ky = 1/4 * torch.fft.fftfreq(Ny, d=1.0/Ny).to(device)
    kz = 1/4 * torch.fft.rfftfreq(Nz, d=1.0/Nz).to(device)  # rfft along last axis

    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')  # [Nx,Ny,Nz//2+1]

    k2 = -(kx**2 + ky**2 + kz**2)
    # linear operator in spectral space
    L = -k2**2 - k2   # |k|^4 - |k|^2
    L = L.to(dtype=torch.float64, device=device)  # real

    # initialize spectral field with rfft
    u_h = torch.fft.rfftn(u0, dim=(-3, -2, -1))  # [B,Nx,Ny,Nz//2+1], complex

    # prepare storage (on CPU)
    record_count = steps // save_every
    actual_records = min(record_count, record_steps)
    sol = torch.zeros(batch, Nx, Ny, Nz, actual_records + 1,
                      dtype=dtype, device='cpu')
    times = torch.zeros(actual_records + 1, dtype=torch.float64)

    # save initial
    sol[..., 0] = u0.detach().cpu()
    times[0] = 0.0
    t = 0.0
    saved = 1

    def nonlinear_spectral(u_phys):
        """
        Nonlinear term N(u) = -0.5 * |∇u|^2
        """
        # transform to spectral
        u_hat = torch.fft.rfftn(u_phys, dim=(-3, -2, -1))
        ux = torch.fft.irfftn(1j * kx * u_hat, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        uy = torch.fft.irfftn(1j * ky * u_hat, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        uz = torch.fft.irfftn(1j * kz * u_hat, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real

        N_phys = -0.5 * (ux**2 + uy**2 + uz**2)
        N_spec = torch.fft.rfftn(N_phys, dim=(-3, -2, -1))
        return N_spec

    # RK4 loop
    for i in range(steps):
        u_phys = torch.fft.irfftn(u_h, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        N1 = nonlinear_spectral(u_phys)
        k1 = dt * (L[None, ...] * u_h + N1)

        u_phys = torch.fft.irfftn(u_h + 0.5 * k1, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        N2 = nonlinear_spectral(u_phys)
        k2 = dt * (L[None, ...] * (u_h + 0.5 * k1) + N2)

        u_phys = torch.fft.irfftn(u_h + 0.5 * k2, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        N3 = nonlinear_spectral(u_phys)
        k3 = dt * (L[None, ...] * (u_h + 0.5 * k2) + N3)

        u_phys = torch.fft.irfftn(u_h + k3, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real
        N4 = nonlinear_spectral(u_phys)
        k4 = dt * (L[None, ...] * (u_h + k3) + N4)

        u_h = u_h + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt

        if (i + 1) % save_every == 0 and saved <= actual_records:
            u_save = torch.fft.irfftn(u_h, s=(Nx, Ny, Nz), dim=(-3, -2, -1)).real.detach()
            sol[..., saved] = u_save.cpu()
            times[saved] = t
            max_amp = u_save.abs().max().item()
            print(f"Step {i+1}/{steps}, t={t:.6f}, max|u|={max_amp:.6e}")
            saved += 1

    return sol
