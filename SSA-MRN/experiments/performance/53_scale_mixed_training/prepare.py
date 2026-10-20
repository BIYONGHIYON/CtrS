"""Coarse (2x lower resolution) training pairs built from the 64x64 training patches only.

(Run as coarse.py under the working number 52; preserved unchanged in provenance/code/coarse.py.)

GT' and PAN' = sensor-MTF blur designed for a x2 reduction (same genMTF/fwind1 construction and QB GNyq values
as common/observation.mtf_kernels and the PAN kernel of common/evaluation_protocol, ratio 2, 41 taps), replicate
padding, decimation at phase 1. MS' = the dataset x4 Wald operator (common/observation.SensorObservation, QB,
phase 2) applied to GT'. LMS' = interp23tap(MS'), the interpolation that reproduces the dataset LMS.
All arrays stay in the normalized DN/2047 coordinates of the training cache.
"""
import numpy as np
import torch
from torch.nn import functional as F

from common.interpolation import interp23tap
from common.observation import GNYQ, SensorObservation

RATIO = 2
SIZE = 41
PAN_GNYQ = 0.15


def gen_mtf(gain, ratio=RATIO, size=SIZE):
    coordinate = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(coordinate, coordinate, indexing='ij')
    window = np.interp(np.sqrt(x * x + y * y), np.arange(size // 2 + 1), np.kaiser(size, 0.5)[size // 2:], right=0.0)
    alpha = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(gain)))
    desired = np.exp(-(x * x + y * y) / (2 * alpha * alpha))
    desired[desired < np.finfo(desired.dtype).eps * desired.max()] = 0
    desired /= desired.max()
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * window)


class CoarsePairs:
    def __init__(self, device='cpu', sensor='QB'):
        self.ms_kernel = torch.from_numpy(np.stack([gen_mtf(g) for g in GNYQ[sensor]])[:, None]).to(device)
        self.pan_kernel = torch.from_numpy(gen_mtf(PAN_GNYQ)[None, None]).to(device)
        self.wald = SensorObservation(sensor, 'decimate', (2, 2)).to(device).double()

    @staticmethod
    def _blur_down(x, kernel):
        pad = kernel.shape[-1] // 2
        rows = torch.arange(-pad, x.shape[-2] + pad, device=x.device).clamp(0, x.shape[-2] - 1)
        cols = torch.arange(-pad, x.shape[-1] + pad, device=x.device).clamp(0, x.shape[-1] - 1)
        y = F.conv2d(x.index_select(-2, rows).index_select(-1, cols), kernel.to(x.dtype), groups=x.shape[1])
        return y[..., 1::RATIO, 1::RATIO]

    @torch.no_grad()
    def __call__(self, pan, gt):
        """pan (N,1,64,64), gt (N,4,64,64) normalized -> pan', lms', ms', gt' (float32, 32/32/8/32)."""
        pan, gt = pan.double(), gt.double()
        gt2 = self._blur_down(gt, self.ms_kernel)
        pan2 = self._blur_down(pan, self.pan_kernel)
        ms2 = self.wald(gt2)
        lms2 = interp23tap(ms2.float()).double()
        return tuple(t.float().contiguous() for t in (pan2, lms2, ms2, gt2))
