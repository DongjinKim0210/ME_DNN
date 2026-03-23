"""Loss functions with variable-length masking."""
import torch


def masked_mse_loss(y_pred, y, mask):
    """MSE loss with per-sample masking for variable-length sequences."""
    assert not torch.isnan(y_pred).any(), "y_pred contains NaN"
    assert not torch.isnan(y).any(), "y contains NaN"
    mask_len = [int(mask[i].sum().item()) for i in range(mask.shape[0])]
    error_sq = (y_pred - y) ** 2
    per_sample = torch.stack([torch.mean(error_sq[i, :, :mask_len[i]]) for i in range(len(mask_len))])
    return torch.mean(per_sample)


# Aliases: both training loops use the same masked MSE loss
custom_loss = masked_mse_loss
denoising_loss = masked_mse_loss
