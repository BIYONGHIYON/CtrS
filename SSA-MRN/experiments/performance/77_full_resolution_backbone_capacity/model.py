"""Larger full-resolution backbone next to the W2 branch.

The original backbone of SSA-MRN is four 64-channel residual blocks at full resolution (backbone(x) = x + the block residuals).
C2MS adds the W2 branch (PixelUnshuffle(2) -> 128 channels x 8 blocks -> PixelShuffle(2)) to its output. Here the full-resolution
path is enlarged:

C2MS_B128D8:       backbone x + 1x1 64->128 -> 8 residual blocks at 128 channels -> 1x1 128->64; W2 128 x 8 (C2MS)
C2MS_B64D8:        backbone = the original 4 blocks + 4 more 64-channel blocks; W2 128 x 8 (C2MS)
C2MS_B64D8_W64D8:  backbone = the original 4 blocks + 4 more 64-channel blocks; W2 64 x 8 (75 C2MS_W64D8)

The original core and the W2 branch are created first in the same random order as 43 C2 (= 53 C2MS) or 75 C2MS_W64D8, so their
initial weights equal those of the same seed; the new backbone layers are created afterwards. The four added 64-channel blocks
start with a zero last convolution, so C2MS_B64D8 and C2MS_B64D8_W64D8 start with exactly the output of the original SSA-MRN.
C2MS_B128D8 replaces the original blocks, so its initial output differs from it (the unused original block attributes are removed)."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp75_model', HERE.parent / '75_parallel_branch_width_depth/model.py')
M75 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M75)
M43, M54 = M75.M43, M75.M54  # evaluate.py builds the 53/57 C2MS controls through model.M54
SPEC = {'C2MS_B128D8': dict(backbone=(128, 8), branch=(128, 8)),
        'C2MS_B64D8': dict(backbone=(64, 8), branch=(128, 8)),
        'C2MS_B64D8_W64D8': dict(backbone=(64, 8), branch=(64, 8))}
ARMS = tuple(SPEC)
SOURCE = {'C2MS_B128D8': 'C2', 'C2MS_B64D8': 'C2', 'C2MS_B64D8_W64D8': 'C2MS_W64D8 (75)'}


class WideBackbone(nn.Module):
    """x + 1x1 stem->width -> depth residual blocks -> 1x1 width->stem; the identity path mirrors the original stacked blocks."""
    def __init__(self, width, depth, stem=64):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(stem, width, 1), *[M43.Block(width) for _ in range(depth)], nn.Conv2d(width, stem, 1))

    def forward(self, x): return x + self.body(x)


def deepened(original, extra):
    blocks = [M43.Block(64) for _ in range(extra)]
    for b in blocks: nn.init.zeros_(b.net[2].weight); nn.init.zeros_(b.net[2].bias)
    return nn.Sequential(*original, *blocks)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        (bw, bd), (ww, wd) = SPEC[arm]['backbone'], SPEC[arm]['branch']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            parallel = M43.Parallel(self.core.backbone) if (ww, wd) == (128, 8) else M75.SizedParallel(self.core.backbone, ww, wd)
            if bw == 64:
                parallel.original = deepened(parallel.original, bd - 4)
            else:
                parallel.original = WideBackbone(bw, bd)
                for name in ('res1', 'res2', 'res3', 'res4'): delattr(self.core, name)
            self.core.backbone = parallel

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)
