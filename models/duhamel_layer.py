"""Duhamel Convolutional Layer: physics-encoded 1D convolution with Mirror-IRF kernels."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuhamelLayer(nn.Module):
    """Duhamel integral layer parameterized by modal frequencies and damping.

    Trainable parameters: log_omegas (log of circular frequencies)
    Fixed: xi_init (damping ratio), dt (time step)
    """
    def __init__(self, omegas, xi_init, dt, uj_u1, max_k=4001):
        super().__init__()
        self.in_channels = 1
        self.out_channels = len(omegas)

        log_omegas = torch.log(torch.tensor(omegas, dtype=torch.float32))
        self.log_omegas = nn.Parameter(log_omegas.clone())
        self.omegas_init = torch.exp(self.log_omegas)
        self.xi_init = torch.tensor(xi_init, dtype=torch.float32)
        self.dt = dt
        self.uj_u1 = uj_u1
        self.max_k = int(max_k)

        self.valid_window_size = [
            int(a) for a in (
                2 * torch.pi / self.omegas_init / torch.sqrt(1 - torch.tensor(xi_init) ** 2)
            ) * (
                1 / (2 * torch.pi * torch.tensor(xi_init)) * torch.log(torch.tensor(1 / self.uj_u1))
            ) / self.dt
        ]
        self.duhamel_kernel_size = [2 * a - 1 for a in self.valid_window_size]

    def forward(self, inputs):
        device = inputs.device
        max_kernel_size = max(self.duhamel_kernel_size)
        raw_omegas = torch.exp(self.log_omegas)
        omegas = torch.clamp(raw_omegas, min=0.01, max=1000)

        IRFlist = []
        for i in range(self.out_channels):
            tt = torch.arange(self.valid_window_size[i], device=device) * self.dt
            omegaD = omegas[i] * torch.sqrt(1 - self.xi_init ** 2)
            IRF_temp = (1 / omegaD) * torch.exp(-self.xi_init * omegas[i] * tt) * torch.sin(omegaD * tt)
            # Mirror-IRF kernel
            weights = torch.cat((IRF_temp.flip(0), torch.zeros(self.duhamel_kernel_size[i] // 2, device=device)))
            addpad = max_kernel_size - self.duhamel_kernel_size[i]
            weights_padded = F.pad(weights, (addpad // 2, addpad // 2), mode='constant', value=0)
            IRFlist.append(weights_padded.view(1, 1, max_kernel_size))
        IRFs = torch.cat(IRFlist, dim=0)

        inputs_pad = F.pad(inputs, (max_kernel_size // 2, max_kernel_size // 2), mode='constant', value=0)
        return F.conv1d(inputs_pad, IRFs, bias=None, stride=1)
