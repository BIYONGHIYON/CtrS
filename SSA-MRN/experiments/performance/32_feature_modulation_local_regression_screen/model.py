"""H1/H2 screening models on the original SSA-MRN (A0) with exact warm start.

W0  A0 unchanged (continued-training control)
W1  H2: PAN-conditioned scale/shift of backbone features after resblock 2 and 4
W1c W1 with the guide computed from backbone features instead of PAN (parameter-matched)
W2  H1: local-regression (guided-filter) prior, fed to the backbone input and blended
    into the LMS base with a per-band gamma (zero initial)
Every new output layer starts at zero, so all arms start exactly at A0.
The original SSA modules, fusion convolutions and decoder are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

ARMS = ('W0', 'W1', 'W1c', 'W2')


@torch.no_grad()
def guided_prior(pan, ms, eps, r=2):
    """Per-band local linear model MS_b ~ a*PAN_lr + c fitted at MS scale, applied to full PAN."""
    h, w = ms.shape[-2:]
    p = F.interpolate(pan, size=(h, w), mode='bicubic', align_corners=False, antialias=True)

    def box(x):
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='reflect'), 2 * r + 1, stride=1)
    mp, mm = box(p), box(ms)
    cov = box(p * ms) - mp * mm
    var = box(p * p) - mp * mp
    a = cov / (var + eps)
    c = mm - a * mp
    a, c = box(a), box(c)
    size = pan.shape[-2:]
    a = F.interpolate(a, size=size, mode='bicubic', align_corners=False)
    c = F.interpolate(c, size=size, mode='bicubic', align_corners=False)
    return a * pan + c


def _zero(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ScreenNet(RestoredPansharpeningNet):
    def __init__(self, arm, eps=1e-4, channels=4):
        super().__init__(channels, 4)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.eps = eps
        if arm == 'W1':
            self.guide = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1))
        if arm == 'W1c':
            self.guide = nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1))
        if arm in ('W1', 'W1c'):
            self.modulation = nn.ModuleList(_zero(nn.Conv2d(128, 128, 1)) for _ in range(2))
        if arm == 'W2':
            self.prior_in = _zero(nn.Conv2d(2 * channels, 64, 3, padding=1, bias=False))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, pan, lms, ms):
        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        y = self.cov2t64(torch.cat([lms, pan, pan_updown], dim=1))
        if self.arm == 'W2':
            gf = guided_prior(pan, ms, self.eps)
            y = y + self.prior_in(torch.cat([gf, gf - lms], dim=1))
        if self.arm in ('W1', 'W1c'):
            g = self.guide(pan if self.arm == 'W1' else y)
        k = 0
        for i, block in enumerate(self.backbone):
            y = block(y)
            if self.arm in ('W1', 'W1c') and i in (1, 3):
                scale, shift = self.modulation[k](torch.cat([y, g], dim=1)).chunk(2, dim=1)
                y = y * (1 + scale) + shift
                k += 1
        res = self.conv64t48(y) + lms
        if self.arm == 'W2':
            res = res + self.gamma * (gf - lms)

        # Below is the original restored forward, unchanged.
        pan_ms = self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res = res + pan_ms
        res_half = self.downsample100(res)
        ms_half = self.upsample100(ms)
        pan_half = self.downsample100(pan)
        pan_ms = self._attend(pan_half, res_half, ms_half, self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        res_half = res_half + pan_ms
        res_quarter = self.downsample200(res_half)
        pan_quarter = self.downsample2(pan)
        pan_ms = self._attend(pan_quarter, res_quarter, ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)
        output = self.conv2ctc1(torch.cat([ms, pan_ms], 1))
        output = self.conv2ctc2(torch.cat([self.upsample101(output), res_half], 1))
        output = self.conv2ctc3(torch.cat([self.upsample102(output), res], 1))
        return output
