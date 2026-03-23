"""Auxiliary layers: MovingAverage1d."""
import torch
import torch.nn as nn


class MovingAverage1d(nn.Module):
    """Depthwise 1D convolution with a fixed uniform kernel for smoothing."""
    def __init__(self, channels, window=11):
        super().__init__()
        kernel = torch.ones(window) / window
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.conv = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=window, padding=window // 2,
            groups=channels, bias=False,
        )
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
