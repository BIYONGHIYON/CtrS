"""Two reduced W2 branches between the C2MS branch (128 channels x 8 blocks) and the matched branch of experiment 74 (64 x 4):
C2MS_W128D4 keeps the 128-channel width with four blocks, C2MS_W64D8 keeps eight blocks at the 64-channel width.

The branch is the C2MS one with width and depth as parameters: PixelUnshuffle(2) of the 64-channel stem feature (256 channels)
-> 1x1 256->width -> depth residual blocks (3x3 conv, ReLU, 3x3 conv, skip) -> 1x1 width->256 (zero init) -> PixelShuffle(2),
added to the original backbone output. The zero last projection makes the initial output equal the original SSA-MRN."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c2ms54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
SHAPES = {'C2MS_W128D4': (128, 4), 'C2MS_W64D8': (64, 8)}
ARMS = tuple(SHAPES)
CORE43 = {arm: 'C0' for arm in ARMS}


class SizedParallel(nn.Module):
    def __init__(self, original, width, depth, stem=64):
        super().__init__(); self.original = original
        self.branch = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * stem, width, 1), *[M43.Block(width) for _ in range(depth)],
                                    nn.Conv2d(width, 4 * stem, 1), nn.PixelShuffle(2))
        nn.init.zeros_(self.branch[-2].weight); nn.init.zeros_(self.branch[-2].bias)

    def forward(self, x): return self.original(x) + self.branch(x)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            self.core.backbone = SizedParallel(self.core.backbone, *SHAPES[arm])

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)
