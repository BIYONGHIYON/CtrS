"""Masked reflectance and spectral-direction reconstruction losses."""
import torch

def reconstruction_loss(prediction, target, valid_mask=None, spectral_weight=0., spectral_eps=1e-6):
    # float32 reductions keep 204-band norms stable under CUDA autocast.
    pred, gt = prediction.float(), target.float()
    mask = torch.ones_like(gt[:, 0], dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if mask.ndim == 4:
        mask = mask[:, 0]
    mse = ((pred - gt).square() * mask[:, None]).sum() / (mask.sum() * gt.shape[1]).clamp_min(1)
    # Norm floor avoids unstable directions for dark spectra in either image.
    pred_norm = torch.linalg.vector_norm(pred, dim=1)
    gt_norm = torch.linalg.vector_norm(gt, dim=1)
    selected = mask & (gt_norm > spectral_eps) & (pred_norm > spectral_eps)
    denom = pred_norm.clamp_min(spectral_eps) * gt_norm.clamp_min(spectral_eps)
    cosine = ((pred * gt).sum(1) / denom).clamp(-1., 1.)
    spectral = ((1. - cosine) * selected).sum() / selected.sum().clamp_min(1)
    return mse + spectral_weight * spectral, mse, spectral
