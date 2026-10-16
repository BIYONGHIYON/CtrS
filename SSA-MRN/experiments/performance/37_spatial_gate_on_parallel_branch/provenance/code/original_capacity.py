import torch
from torch import nn

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
