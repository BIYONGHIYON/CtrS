
import torch
from torch import nn
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class SharedThreeStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = RestoredPansharpeningNet(4, 4)
        self.error_encoder = nn.Conv2d(4, 4, 3, padding=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, pan, lms, ms):
        current = lms
        for _ in range(3):
            conditioned = current + self.error_encoder(torch.zeros_like(current))
            proposal = self.core(pan, conditioned, ms)
            current = current + self.alpha * (proposal - current)
        return current

def build(arm):
    if arm == "B0": return RestoredPansharpeningNet(4, 4)
    if arm == "B1": return SharedThreeStage()
    raise ValueError(arm)

def core(model):
    return model.core if isinstance(model, SharedThreeStage) else model

def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
