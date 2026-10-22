"""Sensor table, x4 Wald operators and 2x coarser training pairs for GF2, WV3 and WV2 (QB kept as a reference).

MS GNyq: DLPan-Toolbox genMTF.m (QB, WV3, WV2; GF2 is not listed, so the toolbox uses its default 0.3 for every band).
PAN GNyq: DLPan-Toolbox MTF_PAN.m (QB 0.15, WV3 0.14, WV2 0.11; GF2 is not listed, the toolbox 'none' value 0.15 is used).
The MS operators were checked against the data (provenance/sensor_audit.json): decimate(MTF(GT)) at phase 2 reproduces the
given MS. The PAN value cannot be checked from the data. Same genMTF/fwind1 construction as common/observation.mtf_kernels.
"""
import numpy as np
import torch
from torch.nn import functional as F

from common.interpolation import interp23tap

SENSORS = {
    'QB': dict(bands=4, max_value=2047., ms_gnyq=[0.34, 0.32, 0.30, 0.22], pan_gnyq=0.15, folder='QuickBird', tag='qb', rgb=[2, 1, 0]),
    'GF2': dict(bands=4, max_value=1023., ms_gnyq=[0.3] * 4, pan_gnyq=0.15, folder='Gaofen 2', tag='gf2', rgb=[2, 1, 0]),
    'WV3': dict(bands=8, max_value=2047., ms_gnyq=[0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], pan_gnyq=0.14, folder='WorldView 3', tag='wv3', rgb=[4, 2, 1]),
    'WV2': dict(bands=8, max_value=2047., ms_gnyq=[0.35] * 7 + [0.27], pan_gnyq=0.11, folder='WorldView 2', tag='wv2', rgb=[4, 2, 1]),
}
PHASE = 2
SIZE = 41


def data_files(sensor):
    s = SENSORS[sensor]; d, t = s['folder'], s['tag']
    return {'train': f'{d}/Training Dataset/train_{t}.h5', 'val': f'{d}/Training Dataset/valid_{t}.h5',
            'RR': f'{d}/Testing Dataset (ReducedData, H5 Format)/test_{t}_multiExm1.h5', 'FR': f'{d}/Testing Dataset (FullData, H5 Format)/test_{t}_OrigScale_multiExm1.h5'}


def gen_mtf(gain, ratio, size=SIZE):
    coordinate = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(coordinate, coordinate, indexing='ij')
    window = np.interp(np.sqrt(x * x + y * y), np.arange(size // 2 + 1), np.kaiser(size, 0.5)[size // 2:], right=0.0)
    alpha = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(gain)))
    desired = np.exp(-(x * x + y * y) / (2 * alpha * alpha))
    desired[desired < np.finfo(desired.dtype).eps * desired.max()] = 0
    desired /= desired.max()
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * window)


def _blur(x, kernel):
    pad = kernel.shape[-1] // 2
    rows = torch.arange(-pad, x.shape[-2] + pad, device=x.device).clamp(0, x.shape[-2] - 1)
    cols = torch.arange(-pad, x.shape[-1] + pad, device=x.device).clamp(0, x.shape[-1] - 1)
    return F.conv2d(x.index_select(-2, rows).index_select(-1, cols), kernel.to(x.dtype), groups=x.shape[1])


class Wald:
    """x4 degradation of a sensor: MS = decimate(MTF_MS(x), phase 2), PAN = decimate(MTF_PAN(pan), phase 2)."""

    def __init__(self, sensor, device='cpu'):
        s = SENSORS[sensor]; self.sensor, self.max_value = sensor, s['max_value']
        self.ms_kernel = torch.from_numpy(np.stack([gen_mtf(g, 4) for g in s['ms_gnyq']])[:, None]).to(device).double()
        self.pan_kernel = torch.from_numpy(gen_mtf(s['pan_gnyq'], 4)[None, None]).to(device).double()

    def ms(self, x): return _blur(x, self.ms_kernel)[..., PHASE::4, PHASE::4]
    def pan(self, x): return _blur(x, self.pan_kernel)[..., PHASE::4, PHASE::4]

    def degrade(self, pan_dn, ms_dn, device):
        """numpy DN (1,H,W), (C,h,w) -> degraded (pan, lms, ms) numpy DN arrays (dataset-consistent RR of the input)."""
        with torch.no_grad():
            p = torch.from_numpy(np.asarray(pan_dn, dtype=np.float64))[None].to(device); m = torch.from_numpy(np.asarray(ms_dn, dtype=np.float64))[None].to(device)
            pan_b, ms_b = self.pan(p), self.ms(m)
            lms_b = interp23tap((ms_b / self.max_value).float()).double() * self.max_value
        return pan_b[0].cpu().numpy(), lms_b[0].cpu().numpy(), ms_b[0].cpu().numpy()

    def observation_rmse(self, fused_dn, ms_dn, device, margin=5):
        with torch.no_grad():
            f = torch.from_numpy(np.asarray(fused_dn, dtype=np.float64))[None].to(device)
            e = self.ms(f)[0].cpu().numpy() - np.asarray(ms_dn, dtype=np.float64)
        return float(np.sqrt(np.mean(e[..., margin:-margin, margin:-margin] ** 2)))


class CoarsePairs:
    """2x coarser pairs from normalized training patches (same construction as 53 prepare.py, sensor-specific MTF)."""

    def __init__(self, sensor, device='cpu'):
        s = SENSORS[sensor]
        self.ms_kernel = torch.from_numpy(np.stack([gen_mtf(g, 2) for g in s['ms_gnyq']])[:, None]).to(device)
        self.pan_kernel = torch.from_numpy(gen_mtf(s['pan_gnyq'], 2)[None, None]).to(device)
        self.wald = Wald(sensor, device)

    @staticmethod
    def _blur_down(x, kernel):
        return _blur(x, kernel)[..., 1::2, 1::2]

    @torch.no_grad()
    def __call__(self, pan, gt):
        """pan (N,1,64,64), gt (N,C,64,64) normalized -> pan', lms', ms', gt' (float32, 32/32/8/32)."""
        pan, gt = pan.double(), gt.double()
        gt2 = self._blur_down(gt, self.ms_kernel); pan2 = self._blur_down(pan, self.pan_kernel)
        ms2 = self.wald.ms(gt2)
        lms2 = interp23tap(ms2.float()).double()
        return tuple(t.float().contiguous() for t in (pan2, lms2, ms2, gt2))
