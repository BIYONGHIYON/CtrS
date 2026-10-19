"""Matched original, parallel-capacity, and independent refinement candidates."""
import copy, importlib
import torch
from torch import nn
from common.model import RestoredPansharpeningNet
I1=importlib.import_module('research.performance.17_structural_ablation.model').StructureNet
RefinementNet=importlib.import_module('research.performance.28_residual_refinement.model').RefinementNet
VARIANTS=('R0','W2','W3','F01','F11')
class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+self.net(x)
class Parallel(nn.Module):
    def __init__(self,original,low):
        super().__init__();self.original=original;self.low=low;c=128 if low else 64;n=8
        self.branch=nn.Sequential(nn.PixelUnshuffle(2) if low else nn.Identity(),nn.Conv2d(256 if low else 64,c,1),*[Block(c) for _ in range(n)],nn.Conv2d(c,256 if low else 64,1),nn.PixelShuffle(2) if low else nn.Identity())
        nn.init.zeros_(self.branch[-2].weight);nn.init.zeros_(self.branch[-2].bias);self.ratio=None
    def forward(self,x):
        b=self.original(x);y=self.branch(x)
        if not self.training:self.ratio=(y.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)).detach()
        return b+y
class StructureNet(nn.Module):
    def __init__(self,variant,channels=4):
        super().__init__();self.variant=variant
        # Reset seed to construct precisely the same original core in every candidate.
        torch.manual_seed(42);base=RestoredPansharpeningNet(4,4)
        if variant in ('F01','F11'):
            self.net=RefinementNet('G5',4);self.net.core.load_state_dict(base.state_dict())
            if variant=='F11':
                core=I1('I1');sd=base.state_dict();sd['cov2t64.weight']=sd['cov2t64.weight'][:,:5].clone();core.load_state_dict(sd);self.net.core=core
        else:
            self.net=base
            if variant in ('W2','W3'):self.net.backbone=Parallel(self.net.backbone,variant=='W2')
    def configure(self,c):
        if self.variant in ('F01','F11'):self.net.configure(c)
    def forward(self,*x):return self.net(*x)
    def gate_values(self):
        b=getattr(self.net,'backbone',None)
        return {'branch_rms_ratio':float(b.ratio)} if isinstance(b,Parallel) and b.ratio is not None else {}
