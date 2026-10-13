"""QB structural screening: full-resolution dual streams and shared feature refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet
VARIANTS=('C0','C1','C2','C3','C4','C5')
class Residual(nn.Module):
    def __init__(self,width):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1))
    def forward(self,x):return x+0.1*self.body(x)
class Exchange(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.projection=nn.Conv2d(width,width,3,padding=1)
        self.mix=nn.Sequential(nn.Conv2d(width*2,width,1),nn.GELU(),Residual(width),Residual(width))
    def forward(self,s,p,difference):
        # Equal parameter count: plain arm processes PAN, difference arm predicts PAN from MS features.
        guide=p-self.projection(s) if difference else self.projection(p)
        return self.mix(torch.cat((s,guide),1))
class DualStreamCore(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.spectral=nn.Sequential(nn.Conv2d(8,width,3,padding=1),nn.GELU(),Residual(width),Residual(width))
        self.spatial=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.GELU(),Residual(width),Residual(width))
        self.exchange=nn.ModuleList([Exchange(width) for _ in range(3)])
        self.context=nn.Sequential(nn.Conv2d(width*3,width,1),nn.GELU(),Residual(width))
        self.head=nn.Sequential(Residual(width),nn.Conv2d(width,4,3,padding=1))
        nn.init.normal_(self.head[-1].weight,std=0.001);nn.init.zeros_(self.head[-1].bias)
    def forward(self,pan,lms,ms,difference=False,steps=1):
        if pan.shape[1]!=1 or lms.shape[1]!=4 or ms.shape[-2:]!=(lms.shape[-2]//4,lms.shape[-1]//4):raise ValueError('QB NCHW / scale4 expected')
        up=F.interpolate(ms,size=lms.shape[-2:],mode='bilinear',align_corners=False)
        s=self.spectral(torch.cat((lms,up),1));p=self.spatial(pan)
        pp=[p,F.avg_pool2d(p,2),F.avg_pool2d(p,4)]
        for _ in range(steps):
            ss=[s,F.avg_pool2d(s,2),F.avg_pool2d(s,4)]
            exchanged=[m(x,y,difference) for m,x,y in zip(self.exchange,ss,pp)]
            full=[exchanged[0]]+[F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False) for x in exchanged[1:]]
            s=s+0.1*self.context(torch.cat(full,1))
        return lms+self.head(s)
class DualStreamNet(nn.Module):
    def __init__(self,variant='C0'):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant
        if variant in ('C0','C1'):
            self.core=RestoredPansharpeningNet(4,4)
            if variant=='C1':
                self.error_encoder=nn.Conv2d(4,4,3,padding=1);self.alpha=nn.Parameter(torch.tensor(0.1))
        else:self.core=DualStreamCore()
    def gate_values(self):return {'alpha':float(self.alpha.detach().cpu())} if self.variant=='C1' else {}
    def forward(self,pan,lms,ms):
        if self.variant=='C0':return self.core(pan,lms,ms)
        if self.variant=='C1':
            current=lms
            for _ in range(3):
                proposal=self.core(pan,current+self.error_encoder(torch.zeros_like(current)),ms)
                current=current+self.alpha*(proposal-current)
            return current
        return self.core(pan,lms,ms,self.variant in ('C3','C5'),2 if self.variant in ('C4','C5') else 1)
