"""C2MS with per-pixel affine modulation in the eight W2 residual blocks (64); C2MS itself for the controls.

C2MS_AM: every W2 block x + net(x) becomes x + net(x) * (1 + m(x)) + b(x), where [m, b] come from a small sub-network
(1x1 128->32, PReLU, 3x3 32->32, PReLU, 1x1 32->256) at the W2 resolution (H/2). This is the per-pixel affine modulation
of ARConv (CVPR 2025), whose removal cost the most SAM in its ablation, applied to the existing W2 convolutions. The
C2MS weights are built exactly as 54 Model('C2MS', bands, seed); the modulation weights are created afterwards under a
separate forked seed and their last conv is zero, so the initial C2MS_AM has the 43 C2 initial weights and output.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c2ms54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C2MS_AM',)
CORE43 = {'C2MS': 'C2', 'C2MS_AM': 'C2'}
MOD_SEED_OFFSET = 1000


class AffineBlock(nn.Module):
    """43 W2 Block with the same `net` parameters plus a per-pixel affine modulation `mod`."""

    def __init__(self, block, c=128, w=32):
        super().__init__()
        self.net = block.net
        self.mod = nn.Sequential(nn.Conv2d(c, w, 1), nn.PReLU(w), nn.Conv2d(w, w, 3, padding=1), nn.PReLU(w), nn.Conv2d(w, 2 * c, 1))
        nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x):
        m, b = self.mod(x).chunk(2, 1)
        return x + self.net(x) * (1 + m) + b


def Model(arm, bands=4, seed=42):
    m = M54.Model('C2MS', bands, seed)
    if arm == 'C2MS': return m
    if arm not in ARMS: raise ValueError(arm)
    branch = m.core.backbone.branch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + MOD_SEED_OFFSET)
        for i, layer in enumerate(branch):
            if isinstance(layer, M43.Block): branch[i] = AffineBlock(layer)
    return m
