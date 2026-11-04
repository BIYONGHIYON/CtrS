"""C2MS with the W2 branch matched to the original backbone: PixelUnshuffle(2) and PixelShuffle(2) are kept, the inside of the
branch becomes the original backbone pipeline (four 64-channel conv-ReLU-conv residual blocks) between 1x1 projections 256->64
and 64->256. The last projection starts at zero, so the initial output equals the original SSA-MRN of the same seed."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c2ms54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C2MS_M',)
CORE43 = {'C2MS_M': 'C0'}


class MatchedParallel(nn.Module):
    def __init__(self, original, channels=64, blocks=4):
        super().__init__(); self.original = original
        self.branch = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * channels, channels, 1), *[M43.Block(channels) for _ in range(blocks)],
                                    nn.Conv2d(channels, 4 * channels, 1), nn.PixelShuffle(2))
        nn.init.zeros_(self.branch[-2].weight); nn.init.zeros_(self.branch[-2].bias)

    def forward(self, x): return self.original(x) + self.branch(x)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            self.core.backbone = MatchedParallel(self.core.backbone)

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)
