"""EnsembleModeDuhamel: Physics-encoded DNN for seismic response prediction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .duhamel_layer import DuhamelLayer
from .layers import MovingAverage1d


class EnsembleModeDuhamel(nn.Module):
    """Full physics-encoded model combining Duhamel, Mode Ensemble, and Reconstruction layers.

    Architecture:
        1. Duhamel Convolutional Layer (modal displacement from ground motion)
        2. Mode Ensemble Layer (mode shapes via QR, mass matrix, MPF)
        3. Reconstruction Layer (modal superposition, disp->acc, MA smoothing)
    """
    def __init__(self, freq_list, dt, xi_init, uj_u1, num_node, device_allocate,
                 ma_window=5):
        super().__init__()
        self.device_allocate = device_allocate
        self.omegas = 2 * np.pi * freq_list
        self.dt = dt
        self.num_node = num_node
        self.num_mode = num_node

        # Displacement to acceleration: 2nd-order central difference
        dsp_to_acc_conv = nn.Conv1d(
            in_channels=self.num_node, out_channels=self.num_node,
            kernel_size=3, padding=1, bias=False, groups=self.num_node,
        )
        kernel = torch.tensor([1 / dt**2, -2 / dt**2, 1 / dt**2], dtype=torch.float32)
        dsp_to_acc_conv.weight.data = kernel.repeat(self.num_node, 1, 1)
        dsp_to_acc_conv.weight.requires_grad = False
        self.dsp_to_acc_conv = dsp_to_acc_conv

        # Duhamel convolutional layer
        self.duhamel_convs = DuhamelLayer(self.omegas, xi_init, self.dt, uj_u1, max_k=4001)

        # Trainable mode shapes (num_mode x num_node)
        self.ModeShapeT_DL = nn.Parameter(torch.randn(self.num_mode, self.num_node, dtype=torch.float32))

        # Trainable diagonal mass matrix
        init_mass = torch.normal(mean=1000.0, std=50.0, size=(self.num_node,))
        self.mass_diag_raw = nn.Parameter(init_mass)

        # Moving average smoothing
        self.MA_filter = MovingAverage1d(channels=num_node, window=ma_window)

    def forward(self, inputtsdata):
        device = self.device_allocate or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        batchsize = inputtsdata.shape[0]
        timesize = inputtsdata.shape[-1]

        self.dsp_to_acc_conv = self.dsp_to_acc_conv.to(device)
        self.ModeShapeT_DL = self.ModeShapeT_DL.to(device)
        self.mass_diag_raw = self.mass_diag_raw.to(device)
        massmatrix = torch.diag(self.mass_diag_raw).to(device)
        self.MA_filter = self.MA_filter.to(device)

        # 1. Duhamel integral → modal displacement
        duhamel_outputs = self.duhamel_convs(inputtsdata) * self.dt

        # 2. Mode shapes via QR decomposition (orthogonality constraint)
        Qtemp, Rsign = torch.linalg.qr(self.ModeShapeT_DL)
        Qsign = Qtemp @ torch.diag(torch.sign(torch.diag(Rsign)))
        modeshapes_n = F.normalize(Qsign, p=2, dim=1)

        # 3. Modal Participation Factor
        mpf_theory = (
            torch.linalg.inv(modeshapes_n @ massmatrix @ modeshapes_n.T)
            @ (modeshapes_n @ massmatrix @ torch.ones(self.num_node, 1, dtype=torch.float32, device=device))
        )

        # 4. MPF-scaled modal displacement → nodal displacement via modal superposition
        modes_dsp_mpf_scaled = -duhamel_outputs * mpf_theory.unsqueeze(0)
        nodal_dsp = torch.einsum('md,bmt->bdt', modeshapes_n, modes_dsp_mpf_scaled)

        # 5. Add ground floor (zero displacement) and compute acceleration
        zero_floor = torch.zeros(batchsize, 1, timesize, dtype=torch.float32, device=device)
        nodal_dsp_with_floor0 = torch.cat([zero_floor, nodal_dsp], dim=1)
        nodal_acc_resp = self.dsp_to_acc_conv(nodal_dsp_with_floor0[:, 1:, :])

        # 6. Moving average smoothing
        x = self.MA_filter(nodal_acc_resp)
        return x, nodal_dsp
