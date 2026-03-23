"""Denoising Autoencoder (DAE) for noisy seismic input preprocessing."""
import torch
import torch.nn as nn


class DenoisingDNN(nn.Module):
    """Conv1d encoder-decoder for denoising ground motion signals."""
    def __init__(self, device_allocate):
        super().__init__()
        self.device_allocate = device_allocate

        self.DN_enc1 = nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=4)
        self.DN_enc2 = nn.Conv1d(32, 32, kernel_size=9, stride=4, padding=3)
        self.DN_dec2 = nn.ConvTranspose1d(32, 32, kernel_size=9, stride=4, padding=3, output_padding=1)
        self.DN_dec1 = nn.ConvTranspose1d(32, 1, kernel_size=11, stride=4, padding=4, output_padding=1)
        self.DN_maxpool1d = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, inputtsdata):
        batchsize = inputtsdata.shape[0]

        # Per-sample normalization
        inputtsdata_normalized = torch.zeros_like(inputtsdata)
        normalized_factor = torch.zeros(batchsize, device=inputtsdata.device)
        for ii in range(batchsize):
            Mm = torch.abs(inputtsdata[ii:ii + 1]).max()
            inputtsdata_normalized[ii:ii + 1] = inputtsdata[ii:ii + 1] / Mm
            normalized_factor[ii] = Mm

        # Encoder
        Xe1 = self.DN_maxpool1d(self.DN_enc1(inputtsdata_normalized))
        Xe2 = self.DN_maxpool1d(self.DN_enc2(Xe1))

        # Decoder
        Xd2 = self.DN_maxpool1d(self.DN_dec2(Xe2))
        Xd1 = self.DN_dec1(Xd2)

        # Denormalization
        outputtsdata = torch.zeros_like(inputtsdata)
        for ii in range(batchsize):
            outputtsdata[ii:ii + 1] = Xd1[ii:ii + 1] * normalized_factor[ii]
        return outputtsdata
