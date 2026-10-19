"""Phase-preserving PAN auxiliary features; only full-resolution SSA changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'provenance/source'))
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet
ARMS=('B0','C0','F1','F2')
class FrequencyCore(RestoredPansharpeningNet):
    def __init__(self,arm):
        super().__init__(4,4);self.arm=arm
        if arm!='B0':
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.PReLU(),nn.Conv2d(4,4,3,padding=1)) for _ in range(2)])
            self.gates=nn.Parameter(torch.zeros(4,2))
    @staticmethod
    def bands(pan):
        # Fixed reflect halo for both training patches and full-scene evaluation.
        pad=16
        x=F.pad(pan,(pad,pad,pad,pad),mode='reflect');h,w=x.shape[-2:]
        fy=torch.fft.fftfreq(h,device=x.device)[:,None];fx=torch.fft.rfftfreq(w,device=x.device)[None,:]
        rho2=fy.square()+fx.square();g1=torch.exp(-rho2/(2*.10**2));g2=torch.exp(-rho2/(2*.25**2))
        z=torch.fft.rfft2(x,norm='ortho')
        return tuple(torch.fft.irfft2(z*mask,s=(h,w),norm='ortho')[...,pad:-pad,pad:-pad] for mask in (g1,g2-g1,1-g2))
    def _attend(self,pan,features,ms,blocks,fusion,activation):
        if self.arm=='B0' or blocks is not self.SSA_blocks:
            return super()._attend(pan,features,ms,blocks,fusion,activation)
        if self.arm=='C0':a,b=pan,pan
        else:
            _,mid,high=self.bands(pan);a,b=(high,high) if self.arm=='F1' else (mid,high)
        aux=[branch(x) for branch,x in zip(self.branches,(a,b))];outputs=[]
        for band,block in enumerate(blocks):
            band_input=torch.cat([features,ms[:,band:band+1]],dim=1)
            guide=block.conv6t6(block.relu(block.conv1t6(pan)))
            guide=guide+self.gates[band,0]*aux[0]+self.gates[band,1]*aux[1]
            projected=block.conv7t6_3(band_input);n,c,h,w=projected.shape
            p=guide.reshape(n,c,-1);q=projected.permute(0,1,3,2).reshape(n,c,-1)
            weights=(p*q).view(n,c,h,w).permute(0,1,3,2).contiguous().view(n,c,-1)
            outputs.append((p*F.softmax(weights,dim=-1)).contiguous().view(n,c,h,w))
        return activation(fusion(torch.cat(outputs,dim=1)))
class Model(nn.Module):
    def __init__(self,arm,seed=42):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.core=FrequencyCore(arm)
    def forward(self,pan,lms,ms):return self.core(pan,lms,ms)
