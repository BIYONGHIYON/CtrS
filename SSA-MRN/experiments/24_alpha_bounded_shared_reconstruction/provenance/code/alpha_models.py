
import hashlib
import math
import torch
from torch import nn
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class SharedThreeStage(nn.Module):
    def __init__(self, bounded=False):
        super().__init__()
        self.core = RestoredPansharpeningNet(4, 4)
        self.error_encoder = nn.Conv2d(4, 4, 3, padding=1, bias=True)
        self.bounded = bounded
        if bounded:
            self.a = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def effective_alpha(self):
        return self.a.sigmoid() if self.bounded else self.alpha

    def raw_alpha(self):
        return self.a if self.bounded else self.alpha

    def forward(self, pan, lms, ms):
        current = lms
        for _ in range(3):
            conditioned = current + self.error_encoder(torch.zeros_like(current))
            proposal = self.core(pan, conditioned, ms)
            current = current + self.effective_alpha() * (proposal - current)
        return current

def build(arm):
    if arm not in ("B1", "B1_alpha_bounded"):
        raise ValueError(arm)
    return SharedThreeStage(arm == "B1_alpha_bounded")

def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def common_hash(model):
    return state_hash({k:v for k,v in model.state_dict().items()
                       if k.startswith(("core.", "error_encoder."))})
