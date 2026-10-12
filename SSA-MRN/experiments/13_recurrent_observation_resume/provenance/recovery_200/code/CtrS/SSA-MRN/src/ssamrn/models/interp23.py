"""Differentiable channelwise toolbox-style 23-tap LMS interpolation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def interp23tap(image, ratio=4):
    if image.ndim != 4 or ratio < 1 or ratio & (ratio-1):
        raise ValueError('Expected NCHW input and power-of-two interpolation ratio')
    # Same coefficients, periodic boundary and insertion phases as metrics._interp23tap.
    half = [.5,.305334091185,0,-.072698593239,0,.021809577942,
            0,-.005192756653,0,.000807762146,0,-.000060081482]
    with torch.autocast(image.device.type, enabled=False):
        image=image.float()
        kernel=image.new_tensor(half[:0:-1]+half)*2
        channels=image.shape[1]
        vertical=kernel.view(1,1,23,1).expand(channels,1,23,1).contiguous()
        horizontal=kernel.view(1,1,1,23).expand(channels,1,1,23).contiguous()
        for stage in range(int(math.log2(ratio))):
            n,c,h,w=image.shape
            up=image.new_zeros(n,c,h*2,w*2)
            offset=1 if stage==0 else 0
            up[:,:,offset::2,offset::2]=image
            rows=torch.arange(-11,h*2+11,device=image.device)%(h*2)
            cols=torch.arange(-11,w*2+11,device=image.device)%(w*2)
            image=F.conv2d(up.index_select(2,rows),vertical,groups=c)
            image=F.conv2d(image.index_select(3,cols),horizontal,groups=c)
    return image


class Interp23(nn.Module):
    def __init__(self,ratio):
        super().__init__();self.ratio=ratio
    def forward(self,x):
        return interp23tap(x,self.ratio)
