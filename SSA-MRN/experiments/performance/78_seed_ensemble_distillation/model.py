"""78 students and teacher.

Students reuse the archived structures unchanged, so their initial weights equal the same-seed controls trained on GT only:
C2MS_B64D8_W64D8_KD = 77 C2MS_B64D8_W64D8 (E3: full-resolution backbone 64 x 8 + W2 64 x 8)
C2MS_W64D8_KD       = 75 C2MS_W64D8       (original backbone 64 x 4 + W2 64 x 8)
The teacher is the mean of six fixed models: 53 C2MS and 65 C2MS_L1S, QB seeds 42-44 (E_ALL of stage 0)."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PERF = HERE.parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M77 = _module('exp77_model', PERF / '77_full_resolution_backbone_capacity/model.py')
M75, M54, M43 = M77.M75, M77.M54, M77.M43  # evaluate.py builds the 53/57 C2MS controls through M54 and the original through M43
STUDENTS = {'C2MS_B64D8_W64D8_KD': ('77', 'C2MS_B64D8_W64D8'), 'C2MS_W64D8_KD': ('75', 'C2MS_W64D8')}
ARMS = tuple(STUDENTS)


def base_model(arm, bands=4, seed=42):
    """The same-structure control model (77 E3 or 75 W64D8) built with the seed."""
    source, name = STUDENTS[arm]
    return (M77 if source == '77' else M75).Model(name, bands, seed)


class Model(nn.Module):
    def __init__(self, arm, bands=4, seed=42):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm, self.bands = arm, bands
        self.core = base_model(arm, bands, seed).core  # same parameter names (core.*) as the controls, so their best.pt loads strictly

    def forward(self, pan, lms, ms):
        return self.core(pan, lms, ms)


class Teacher(nn.Module):
    """Mean of the six fixed teacher outputs (FP32)."""
    def __init__(self, checkpoints, bands=4):
        super().__init__()
        nets = []
        for path in checkpoints:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            net = M54.Model('C2MS', bands, ck['seed']); net.load_state_dict(ck['model'], strict=True); nets.append(net.eval())
        self.nets = nn.ModuleList(nets)

    def forward(self, pan, lms, ms):
        return torch.stack([n(pan, lms, ms).float() for n in self.nets]).mean(0)


def conv_macs(model, shapes):
    """Multiply-accumulates of every Conv2d for one forward pass (bias, activations, attention and element-wise ops excluded)."""
    total = [0]
    def hook(m, _i, o): total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    try:
        p = next(model.parameters())
        with torch.inference_mode(): model.eval()(*[torch.zeros(s, device=p.device) for s in shapes])
    finally:
        for h in hs: h.remove()
    return total[0]
