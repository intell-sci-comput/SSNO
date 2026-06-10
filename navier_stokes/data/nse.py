import torch
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GaussianRF(object):

    def __init__(self, size, L1=2 * math.pi, L2=2 * math.pi, alpha=2.5, tau=7.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

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

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
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

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N/2.0)

    # Number of steps to final time
    steps = math.ceil(T/delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Negative Laplacian in Fourier space
    lap = (k_x**2 + k_y**2)
    lap[0,0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps)
    sol_t = torch.zeros(record_steps)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        # Velocity field in y-direction = -psi_x
        v = - k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        # Partial x of vorticity
        w_x = k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        # Partial y of vorticity
        w_y = k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        # Dealias
        F_h = dealias* F_h

        # Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc.reshape(-1, 1, 1)*lap[None,...])*w_h)/(1.0 + 0.5*delta_t*visc.reshape(-1, 1, 1)*lap[None,...])

        # Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            # Record solution and time
            sol[...,c] = w.to('cpu')
            sol_t[c] = t

            c += 1
    return sol, sol_t
