"""79 arms: C2MS with a deeper, narrower W2 branch, trained with the 65 L1 + spectral loss (train.py).

C2MS_W64D12_L1S: original SSA-MRN K4 + W2 branch PixelUnshuffle(2) -> 1x1 256->64 -> 12 residual blocks at 64 channels -> 1x1 64->256 ->
                 PixelShuffle(2) (75 SizedParallel; 1,346,014 parameters, the same count as 77 E3; 34.94 GMAC at 256x256)
C2MS_W64D16_L1S: the same with 16 blocks (1,641,438 parameters; 39.77 GMAC)
Built exactly like 75 Model (core first, then the branch, inside one forked RNG), so every layer outside the branch equals 43 C0 of the same
seed and the branch's last 1x1 starts at zero: the initial output equals the original SSA-MRN."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp75_model', HERE.parent / '75_parallel_branch_width_depth/model.py')
M75 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M75)
M43, M54 = M75.M43, M75.M54  # evaluate.py builds the 53 C2MS / 65 C2MS_L1S controls through M54 and the original through M43
SHAPES = {'C2MS_W64D12_L1S': (64, 12), 'C2MS_W64D16_L1S': (64, 16)}
ARMS = tuple(SHAPES)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.core = M43.RestoredPansharpeningNet(bands, 4)
            self.core.backbone = M75.SizedParallel(self.core.backbone, *SHAPES[arm])

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)


def conv_macs(model, shapes):
    """Multiply-accumulates of every Conv2d for one forward pass (bias, activations, attention and element-wise ops excluded; as 78)."""
    total = [0]
    def hook(m, _i, o): total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    try:
        p = next(model.parameters())
        with torch.inference_mode(): model.eval()(*[torch.zeros(s, device=p.device) for s in shapes])
    finally:
        for h in hs: h.remove()
    return total[0]


def used_parameters(model):
    """Parameters that forward() reaches (68: the official code keeps unused layers; they do not change the output)."""
    seen = set(); total = [0]
    def hook(m, _i, _o):
        for p in m.parameters(recurse=False):
            if id(p) not in seen: seen.add(id(p)); total[0] += p.numel()
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        p = next(model.parameters()); b = model.bands if hasattr(model, 'bands') else 4
        with torch.inference_mode(): model.eval()(*[torch.zeros(s, device=p.device) for s in [(1, 1, 64, 64), (1, b, 64, 64), (1, b, 16, 16)]])
    finally:
        for h in hs: h.remove()
    return total[0]
