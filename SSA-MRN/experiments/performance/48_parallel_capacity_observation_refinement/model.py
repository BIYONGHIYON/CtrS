"""43 C0/C2 cores with the 28 G5 observation-error refiners (F1 = C0+G5, WF = C2+G5)."""
import importlib.util, json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
ROOT = PERF.parents[1]
ARMS = ('C0', 'C2', 'F1', 'WF')
CORE = {'C0': 'C0', 'C2': 'C2', 'F1': 'C0', 'WF': 'C2'}
REFINER_SEED_OFFSET = 1000  # refiners of F1 and WF share one initialization per seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# 43 model inserts its preserved source (common/model.py, upstream network.py) on sys.path first.
M43 = _load('exp43_model', PERF / '43_normalized_parallel_capacity/model.py')
M28 = _load('exp28_model', ROOT / 'research/performance/28_residual_refinement/model.py')
CALIBRATION = json.loads((PERF / '29_capacity_refinement/calibration.json').read_text(encoding='utf8'))
Parallel = M43.Parallel


class Model(nn.Module):
    def __init__(self, arm, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm = arm
        # Same RNG path as 43 Model(C0|C2, seed): identical core (and W2 branch) initial weights.
        self.core = M43.Model(CORE[arm], seed).core
        if arm in ('F1', 'WF'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + REFINER_SEED_OFFSET)
                g5 = M28.RefinementNet('G5', 4)
            g5.configure(CALIBRATION)
            self.refiners = g5.refiners  # last 1x1 fuse conv is zero: initial output equals the core output
            for name in ('pan_weights', 'pan_bias', 'blur_kernel'):
                self.register_buffer(name, getattr(g5, name).detach().clone())

    def down(self, x, size):  # 28 RefinementNet.down
        p = self.blur_kernel.shape[-1] // 2
        if p: x = F.conv2d(F.pad(x, (p, p, p, p), mode='reflect'), self.blur_kernel, groups=x.shape[1])
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, pan, lms, ms):
        x = self.core(pan, lms, ms)
        if self.arm in ('C0', 'C2'): return x
        for i in range(2):  # 28 RefinementNet.forward, variant G5
            em = F.interpolate(ms - self.down(x, ms.shape[-2:]), size=x.shape[-2:], mode='bilinear', align_corners=True)
            ep = pan - (x * self.pan_weights).sum(1, keepdim=True) - self.pan_bias
            ep = ep - F.avg_pool2d(F.pad(ep, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
            x = x + self.refiners[i](torch.cat((x, lms, pan, em, ep), 1))
        return x
