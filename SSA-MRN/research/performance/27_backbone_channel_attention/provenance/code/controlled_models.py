import torch
from torch import nn
from types import SimpleNamespace
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class CABlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv20=original.conv20; self.conv21=original.conv21; self.relu=original.relu
        self.ca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(64,4,1),nn.ReLU(),nn.Conv2d(4,64,1))
        nn.init.zeros_(self.ca[-1].weight); nn.init.zeros_(self.ca[-1].bias)
    def forward(self,x):
        r=self.conv21(self.relu(self.conv20(x)))
        return x+2*torch.sigmoid(self.ca(r))*r

def build(arm):
    model=RestoredPansharpeningNet(4,4,1)
    if arm=="E1":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1042)
            for i in range(4):
                b=CABlock(model.backbone[i]); setattr(model,f"res{i+1}",b); model.backbone[i]=b
    elif arm!="E0": raise ValueError(arm)
    return model

def core(model):
    return SimpleNamespace(state_dict=lambda:{k:v for k,v in model.state_dict().items() if ".ca." not in k})

def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
