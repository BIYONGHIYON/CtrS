"""N1 x W2 factorial; unchanged active original core and 29 W2 branch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'provenance/source'))
import torch
from torch import nn
from common.model import RestoredPansharpeningNet
ARMS=('C0','C1','C2','C3')
class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+self.net(x)
class Parallel(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        self.branch=nn.Sequential(nn.PixelUnshuffle(2),nn.Conv2d(256,128,1),*[Block(128) for _ in range(8)],nn.Conv2d(128,256,1),nn.PixelShuffle(2))
        nn.init.zeros_(self.branch[-2].weight);nn.init.zeros_(self.branch[-2].bias)
    def forward(self,x):return self.original(x)+self.branch(x)
class Model(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.core=RestoredPansharpeningNet(4,4)
            if arm in ('C2','C3'):self.core.backbone=Parallel(self.core.backbone)
    def forward(self,pan,lms,ms):
        if self.arm in ('C0','C2'):return self.core(pan,lms,ms)
        with torch.autocast(device_type=ms.device.type,enabled=False):
            variance,mean=torch.var_mean(ms.float(),dim=(1,2,3),correction=0,keepdim=True)
            scale=variance.clamp_min(0).sqrt().clamp_min(.01)
            inputs=tuple((x.float()-mean)/scale for x in (pan,lms,ms))
        y=self.core(*inputs)
        with torch.autocast(device_type=ms.device.type,enabled=False):return y.float()*scale+mean
