import torch
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GaussianRF(object):

    def __init__(self, size, alpha=4, tau=8.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        s1, s2 = size, size
        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = 0.5 * tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*((k1**2 + k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

    def __call__(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = torch.fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))

        if self.mean is not None:
            u += self.mean
        
        return u 


def ks_2d_rk4(u0, T, dt=1e-3, record_steps=100):
    """
    2D Kuramoto-Sivashinsky equation solver using spectral method + RK4 time stepping.
    ∂u/∂t + 1/2 * (∇u)^2 + Δu + Δ^2 u = 0

    Parameters:
        u0: [batch, N, N] tensor, initial condition in physical space
        T: total simulation time
        dt: time step
        record_steps: number of saved steps (default 100)
    Returns:
        sol: [N, N, record_steps] tensor of u in physical space
        times: [record_steps] tensor of time values
    """
    device = u0.device
    batch = u0.shape[0]
    N = u0.shape[-1]
    steps = math.ceil(T / dt)
    save_every = steps // record_steps

    # Wavenumbers
    k_max = N/2
    kx = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    ky = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    # Spectral operators
    lap = -(kx ** 2 + ky ** 2)
    lap2 = lap ** 2
    L = lap + lap2  # Linear operator
    L = - L[None, :] 

    # Initialize
    u_h = torch.fft.fft2(u0)
    sol = torch.zeros(batch, N, N, record_steps+1, device='cpu')
    sol[..., 0] = u0
    t = 0.0
    c = 1

    def nonlinear(u_phys):
        dealias = torch.unsqueeze(torch.logical_and(torch.abs(ky) <= (2.0/3.0)*k_max, torch.abs(kx) <= (2.0/3.0)*k_max).float(), 0)[None, :]
        ux = torch.fft.ifft2(1j * kx * torch.fft.fft2(u_phys)).real
        uy = torch.fft.ifft2(1j * ky * torch.fft.fft2(u_phys)).real
        return -0.5 * torch.fft.fft2(ux ** 2 + uy ** 2) * dealias

    for i in range(steps):
        u_phys = torch.fft.ifft2(u_h).real
        N1 = nonlinear(u_phys)
        k1 = dt * (L * u_h + N1)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k1).real
        N2 = nonlinear(u_phys)
        k2 = dt * (L * (u_h + 0.5 * k1) + N2)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k2).real
        N3 = nonlinear(u_phys)
        k3 = dt * (L * (u_h + 0.5 * k2) + N3)

        u_phys = torch.fft.ifft2(u_h + k3).real
        N4 = nonlinear(u_phys)
        k4 = dt * (L * (u_h + k3) + N4)

        u_h = u_h + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt

        if (i + 1) % save_every == 0:
            sol[:, :, :, c] = torch.fft.ifft2(u_h).real.cpu()
            print(f"Step {i + 1}/{steps}, t={t:.4f}, max(u)={sol[:,:,:,c].max():.4f}")
            c += 1

    return sol
