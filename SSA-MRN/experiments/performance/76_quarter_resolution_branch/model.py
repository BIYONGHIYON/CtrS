"""A third, quarter-resolution parallel branch (W4) next to the original backbone and the W2 branch.

W4: PixelUnshuffle(4) of the 64-channel stem feature (1024 channels at H/4) -> 1x1 1024->64 -> 4 residual blocks at 64 channels
(3x3 conv, ReLU, 3x3 conv, skip) -> 1x1 64->1024 (zero init) -> PixelShuffle(4), added to the output of the two existing paths.

C2MS_W64D8_W4: original backbone + the 64 x 8 W2 branch of experiment 75 + W4.
C2MS_W4: original backbone + the C2MS W2 branch (128 x 8, built exactly as 43 C2 / 53 C2MS) + W4.
The original core and the W2 branch are created first in the same random order as 75 C2MS_W64D8 and 43 C2, so their initial weights
equal those of the same seed; W4 is created afterwards. The zero last projections make the initial output equal the original SSA-MRN."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp75_model', HERE.parent / '75_parallel_branch_width_depth/model.py')
M75 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M75)
M43 = M75.M43
M54 = M75.M54  # evaluate.py builds the 53/57 C2MS controls through model.M54 (alias added after training, before evaluation)
QUARTER = dict(width=64, depth=4)
ARMS = ('C2MS_W64D8_W4', 'C2MS_W4')
INNER = {'C2MS_W64D8_W4': 'C2MS_W64D8 (75)', 'C2MS_W4': 'C2 (43) = C2MS (53)'}


class WithQuarter(nn.Module):
    def __init__(self, inner, width=64, depth=4, stem=64):
        super().__init__(); self.inner = inner
        self.quarter = nn.Sequential(nn.PixelUnshuffle(4), nn.Conv2d(16 * stem, width, 1), *[M43.Block(width) for _ in range(depth)],
                                     nn.Conv2d(width, 16 * stem, 1), nn.PixelShuffle(4))
        nn.init.zeros_(self.quarter[-2].weight); nn.init.zeros_(self.quarter[-2].bias)

    def forward(self, x): return self.inner(x) + self.quarter(x)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            inner = M43.Parallel(self.core.backbone) if arm == 'C2MS_W4' else M75.SizedParallel(self.core.backbone, *M75.SHAPES['C2MS_W64D8'])
            self.core.backbone = WithQuarter(inner, **QUARTER)

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)
