"""Spatial differences of HSI, restricted to pairs of valid pixels."""
import torch


def masked_gradient_l1(prediction, target, mask=None):
    error = prediction.float() - target.float()
    if mask is None:
        mask = torch.ones_like(error[:, :1], dtype=torch.bool)
    else:
        mask = mask.bool()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    horizontal = mask[..., :, 1:] & mask[..., :, :-1]
    vertical = mask[..., 1:, :] & mask[..., :-1, :]
    dx = error[..., :, 1:] - error[..., :, :-1]
    dy = error[..., 1:, :] - error[..., :-1, :]
    numerator = (dx.abs() * horizontal).sum() + (dy.abs() * vertical).sum()
    denominator = ((horizontal.sum() + vertical.sum()) * error.shape[1]).clamp_min(1)
    return numerator / denominator
