"""61 phase 1: a small head on top of the frozen C2MS output that changes only the band ratio of its high band.

y = C2MS(pan, lms, ms); L = per-band sensor MS MTF blur (41 taps, replicate border, as experiment 60 phase 0); H = y - L(y).
Band-ratio head (H1, H2): y' = L + |H| * normalize(r + delta_perp), r = H/|H|, delta_perp = delta - (delta . r) r.
Free residual head (H3): y' = y + delta. The last conv is zero-initialized, so every arm starts exactly at y.
Head inputs: r, u = L/|L|, LMS/|LMS|, |H|/|L| and the PAN high band (PAN minus its PAN MTF blur).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


P54 = _module('exp54_prepare', HERE.parent / '54_multi_sensor_scale_mixed_training/prepare.py')
M54 = _module('exp54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
SENSORS = P54.SENSORS
ARMS = ('H1', 'H2', 'H3')
EPS = 1e-6


def C2MS(sensor, seed):
    return M54.Model('C2MS', SENSORS[sensor]['bands'], seed)


class Blur(nn.Module):
    """Fixed depthwise sensor MTF blur at full resolution (replicate border), float32."""

    def __init__(self, gnyq):
        super().__init__()
        k = np.stack([P54.gen_mtf(g, 4) for g in gnyq])[:, None]
        self.register_buffer('kernel', torch.from_numpy(k).float(), persistent=False)

    def forward(self, x):
        pad = self.kernel.shape[-1] // 2
        return F.conv2d(F.pad(x, (pad,) * 4, mode='replicate'), self.kernel, groups=x.shape[1])


def unit(x):
    n = x.norm(dim=1, keepdim=True)
    return x / n.clamp_min(EPS), n


class Head(nn.Module):
    def __init__(self, arm, sensor, seed, width=32):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        b = SENSORS[sensor]['bands']; self.arm = arm
        self.ms_blur = Blur(SENSORS[sensor]['ms_gnyq']); self.pan_blur = Blur([SENSORS[sensor]['pan_gnyq']])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.body = nn.Sequential(nn.Conv2d(3 * b + 2, width, 3, padding=1), nn.PReLU(width),
                                      nn.Conv2d(width, width, 3, padding=1), nn.PReLU(width),
                                      nn.Conv2d(width, width, 3, padding=1), nn.PReLU(width),
                                      nn.Conv2d(width, b, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)

    def forward(self, y, pan, lms):
        low = self.ms_blur(y); high = y - low
        r, strength = unit(high); u, low_norm = unit(low)
        x = torch.cat([r, u, unit(lms)[0], strength / low_norm.clamp_min(EPS), pan - self.pan_blur(pan)], 1)
        delta = self.body(x)
        if self.arm == 'H3': return y + delta
        delta = delta - (delta * r).sum(1, keepdim=True) * r
        return low + strength * unit(r + delta)[0]


def angle(p, g):
    """Mean per-pixel angle in radians (atan2 form, stable near 0); pixels with zero GT norm are skipped."""
    gh, gn = unit(g); par = (p * gh).sum(1); perp = (p - par.unsqueeze(1) * gh).square().sum(1).add(1e-12).sqrt()
    mask = (gn.squeeze(1) > EPS) & (p.norm(dim=1) > EPS)
    return torch.atan2(perp, par)[mask].mean()
