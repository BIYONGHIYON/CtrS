"""54 arms for any band count: C0 (original SSA-MRN K4) and C2MS (original + W2 parallel branch, the 53 baseline).

Both are built exactly like 43 Model('C0'|'C2', seed) with the band count of the sensor, so on 4-band sensors the initial
weights equal the 43 C0/C2 of the same seed. Only C2MS is trained with the 2x coarser pairs (see train.py)."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp43_model', HERE.parent / '43_normalized_parallel_capacity/model.py')
M43 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M43)
ARMS = ('C0', 'C2MS')
CORE43 = {'C0': 'C0', 'C2MS': 'C2'}
Parallel = M43.Parallel


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            if arm == 'C2MS': self.core.backbone = Parallel(self.core.backbone)

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)
