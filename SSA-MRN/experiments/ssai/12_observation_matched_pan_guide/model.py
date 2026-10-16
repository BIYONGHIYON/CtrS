"""SSAI 8 N1 block with an observation-matched, band-wise PAN guide.

Only the PAN tensor given to each band's SSA block changes:
    PAN*_b = mu_b + g_b * (PAN_stage - mu_P),  g_b = min(s_b / sqrt(s_P^2 + eps^2), g_max)
mu_b, s_b come from the observed LR MS band and mu_P, s_P from PAN area-averaged to the
MS grid, so the guide is recomputed from each observation (RR or FR). No parameters are
added; with use_matched=False the forward equals experiment 8 N1 exactly.
"""
import importlib

import torch
import torch.nn.functional as F

MeanNormalizedNet = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model').MeanNormalizedNet

RATIO = 4
WINDOW = 5


def box(x, window=WINDOW):
    return F.avg_pool2d(F.pad(x, (window // 2,) * 4, mode='reflect'), window, stride=1)


def observation_statistics(pan, ms, local=True):
    """Return (mu_b, s_b, mu_p, s_p) on the MS grid; pan is full resolution."""
    if pan.shape[-2] != ms.shape[-2] * RATIO or pan.shape[-1] != ms.shape[-1] * RATIO:
        raise ValueError('PAN must be exactly 4x the MS grid')
    pan_lr = F.avg_pool2d(pan, RATIO)
    if local:
        if min(ms.shape[-2:]) <= WINDOW // 2:
            raise ValueError('MS grid too small for the reflect-padded local window')
        mu_b = box(ms); mu_p = box(pan_lr)
        s_b = (box(ms * ms) - mu_b * mu_b).clamp_min(0).sqrt()
        s_p = (box(pan_lr * pan_lr) - mu_p * mu_p).clamp_min(0).sqrt()
    else:
        mu_b = ms.mean((-2, -1), keepdim=True); s_b = ms.std((-2, -1), unbiased=False, keepdim=True)
        mu_p = pan_lr.mean((-2, -1), keepdim=True); s_p = pan_lr.std((-2, -1), unbiased=False, keepdim=True)
    return mu_b, s_b, mu_p, s_p


def matched_guide(pan_stage, pan, ms, eps, gain_max, local=True):
    """Band-wise matched PAN at the stage resolution: N x C x h x w (no gradient path)."""
    with torch.no_grad():
        mu_b, s_b, mu_p, s_p = observation_statistics(pan, ms, local)
        gain = (s_b / (s_p * s_p + eps * eps).sqrt()).clamp(max=gain_max)
        size = pan_stage.shape[-2:]

        def resize(t):
            if t.shape[-2:] == size or t.shape[-2:] == (1, 1):
                return t
            return F.interpolate(t, size=size, mode='bilinear', align_corners=False)

        return resize(mu_b) + resize(gain) * (pan_stage - resize(mu_p))


class ObservationMatchedNet(MeanNormalizedNet):
    """B2: local 5x5 (MS grid) statistics. B3: whole-input statistics (diagnostic)."""

    def __init__(self, channels=4, ssai_dimension=4, local=True, eps=None, gain_max=None):
        super().__init__(channels, ssai_dimension, learn_gamma=False)
        self.local = local
        self.eps = eps
        self.gain_max = gain_max
        self.use_matched = True

    def guides(self, pan_stage, pan, ms):
        if not self.use_matched:
            return [pan_stage] * self.channels
        if self.eps is None or self.gain_max is None:
            raise RuntimeError('eps and gain_max must be frozen from train statistics')
        guide = matched_guide(pan_stage, pan, ms, self.eps, self.gain_max, self.local)
        return [guide[:, band:band + 1] for band in range(self.channels)]

    def _attend_guided(self, guides, features, ms, blocks, fusion, activation):
        outputs = []
        for band in range(self.channels):
            band_input = torch.cat([features, ms[:, band:band + 1]], dim=1)
            outputs.append(blocks[band](guides[band], band_input))
        return activation(fusion(torch.cat(outputs, dim=1)))

    def forward(self, pan, lms, ms):
        # Same operation order as RestoredPansharpeningNet.forward; only the SSA guide differs.
        if pan.ndim != 4 or lms.ndim != 4 or ms.ndim != 4:
            raise ValueError('Expected NCHW tensors for pan, lms, and ms')
        n, channels, h, w = lms.shape
        if channels != self.channels or h % 4 or w % 4 or min(h, w) < 4:
            raise ValueError(f'Invalid LMS shape: {tuple(lms.shape)}')
        if pan.shape != (n, self.guide_channels, h, w) or ms.shape != (n, channels, h // 4, w // 4):
            raise ValueError('Expected guide at LMS resolution and MS at 1/4 resolution')

        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        features = torch.cat([lms, pan, pan_updown], dim=1)
        features = self.backbone(self.cov2t64(features))
        res = self.conv64t48(features) + lms

        pan_ms = self._attend_guided(self.guides(pan, pan, ms), res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res = res + pan_ms

        res_half = self.downsample100(res)
        ms_half = self.upsample100(ms)
        pan_half = self.downsample100(pan)
        pan_ms = self._attend_guided(self.guides(pan_half, pan, ms), res_half, ms_half, self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        res_half = res_half + pan_ms

        res_quarter = self.downsample200(res_half)
        pan_quarter = self.downsample2(pan)
        pan_ms = self._attend_guided(self.guides(pan_quarter, pan, ms), res_quarter, ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)

        output = self.conv2ctc1(torch.cat([ms, pan_ms], dim=1))
        output = self.conv2ctc2(torch.cat([self.upsample101(output), res_half], dim=1))
        output = self.conv2ctc3(torch.cat([self.upsample102(output), res], dim=1))
        return output
