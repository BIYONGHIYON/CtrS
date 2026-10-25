"""62: band-ratio heads on the frozen C2MS, 2x2 factorial of C2MS backbone features (F2, F3) and receptive field (F1, F3).

Every arm keeps the 61 output form y' = L + |H| * normalize(r + delta_perp) with H = y - L(y) (sensor MTF lowpass L) and
the 61 head inputs (r, L/|L|, LMS/|LMS|, |H|/|L|, PAN high band). F2/F3 also get the 64-channel output of the C2MS
backbone (original + W2 branch, full resolution), read with a forward hook. The last conv is zero-initialized, so every
arm starts exactly at the C2MS output. The 61 building blocks are imported from 61 model.py.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp61_model', HERE.parent / '61_high_frequency_band_ratio_correction/model.py')
M61 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M61)
SENSORS, Blur, unit, EPS = M61.SENSORS, M61.Blur, M61.unit, M61.EPS
ARMS = {'F0': (False, 'local'), 'F1': (False, 'wide'), 'F2': (True, 'local'), 'F3': (True, 'wide')}
FEATURES = 64


class C2MSWithFeatures(nn.Module):
    """Frozen C2MS returning (output, backbone output) in one forward."""

    def __init__(self, sensor, seed):
        super().__init__(); self.net = M61.C2MS(sensor, seed); self._feat = None
        self.net.core.backbone.register_forward_hook(lambda m, i, o: setattr(self, '_feat', o))

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms); f = self._feat; self._feat = None
        return y, f


def conv(i, o, stride=1): return nn.Conv2d(i, o, 3, stride=stride, padding=1)


class Local(nn.Module):
    """61 head body: 3x3 conv x4, 32 channels (receptive field 9x9)."""

    def __init__(self, cin, cout, w=32):
        super().__init__()
        self.body = nn.Sequential(conv(cin, w), nn.PReLU(w), conv(w, w), nn.PReLU(w), conv(w, w), nn.PReLU(w), conv(w, cout))
        self.last = self.body[-1]

    def forward(self, x): return self.body(x)


class Wide(nn.Module):
    """2-level U-Net 32/48/64, stride-2 convs down, bilinear up, additive skips (receptive field 31x31)."""

    def __init__(self, cin, cout, w=(32, 48, 64)):
        super().__init__()
        a, b, c = w
        self.e0 = nn.Sequential(conv(cin, a), nn.PReLU(a), conv(a, a), nn.PReLU(a))
        self.e1 = nn.Sequential(conv(a, b, 2), nn.PReLU(b), conv(b, b), nn.PReLU(b))
        self.e2 = nn.Sequential(conv(b, c, 2), nn.PReLU(c), conv(c, c), nn.PReLU(c))
        self.u1 = nn.Sequential(conv(c, b), nn.PReLU(b)); self.u0 = nn.Sequential(conv(b, a), nn.PReLU(a))
        self.last = conv(a, cout)

    def forward(self, x):
        x0 = self.e0(x); x1 = self.e1(x0); x2 = self.e2(x1)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.u1(up(x2, x1)) + x1; d0 = self.u0(up(d1, x0)) + x0
        return self.last(d0)


class Head(nn.Module):
    def __init__(self, arm, sensor, seed):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, (self.use_features, kind) = arm, ARMS[arm]; b = SENSORS[sensor]['bands']
        self.ms_blur = Blur(SENSORS[sensor]['ms_gnyq']); self.pan_blur = Blur([SENSORS[sensor]['pan_gnyq']])
        cin = 3 * b + 2 + (FEATURES if self.use_features else 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.body = (Local if kind == 'local' else Wide)(cin, b)
        nn.init.zeros_(self.body.last.weight); nn.init.zeros_(self.body.last.bias)

    def forward(self, y, pan, lms, feat=None):
        low = self.ms_blur(y); high = y - low
        r, strength = unit(high); u, low_norm = unit(low)
        x = [r, u, unit(lms)[0], strength / low_norm.clamp_min(EPS), pan - self.pan_blur(pan)]
        if self.use_features: x.append(feat)
        delta = self.body(torch.cat(x, 1))
        delta = delta - (delta * r).sum(1, keepdim=True) * r
        return low + strength * unit(r + delta)[0]


class Full(nn.Module):
    """C2MS (+ head) taking (pan, lms, ms); head None gives the control C2MS."""

    def __init__(self, c2ms, head=None):
        super().__init__(); self.c2ms, self.head = c2ms, head

    def forward(self, pan, lms, ms):
        y, f = self.c2ms(pan, lms, ms)
        return y if self.head is None else self.head(y, pan, lms, f)
