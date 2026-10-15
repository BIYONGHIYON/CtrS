"""Independent FFL formulation following ICCV2021 default weighting settings.
Orthonormal FFT, alpha1, patch_factor1, no log or batch-averaged spectrum.
Weights are per sample/band and detached. Denominator-zero weights are zero.
Reference: https://github.com/EndlessSora/focal-frequency-loss
"""
import torch

def ffl_components(prediction,target):
    if prediction.ndim!=4 or prediction.shape!=target.shape:
        raise ValueError('FFL expects matching NCHW tensors')
    if not prediction.numel():raise ValueError('Empty FFL input')
    difference=torch.fft.fft2(prediction.float(),norm='ortho')-torch.fft.fft2(target.float(),norm='ortho')
    squared=difference.real.square()+difference.imag.square()
    magnitude=squared.detach().sqrt()
    maximum=magnitude.amax(dim=(-2,-1),keepdim=True)
    # Match the official zero/NaN normalization behavior without generating NaNs.
    denominator=torch.where(maximum>0,maximum,torch.ones_like(maximum))
    weights=(magnitude/denominator).clamp(0,1).detach()
    return (weights*squared).sum(),torch.tensor(squared.numel(),device=squared.device,dtype=torch.int64)

def ffl_mean(prediction,target):
    total,count=ffl_components(prediction,target)
    return total/count
