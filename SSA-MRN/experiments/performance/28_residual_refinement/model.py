"""Original SSA-MRN followed by two full-resolution residual corrections."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

class Refiner(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3*c+2,32,3,padding=1),nn.PReLU(),nn.Conv2d(32,32,3,padding=1),nn.PReLU())
        self.fuse=nn.Sequential(nn.Conv2d(96,32,1),nn.PReLU(),nn.Conv2d(32,c,3,padding=1))
        nn.init.zeros_(self.fuse[-1].weight);nn.init.zeros_(self.fuse[-1].bias)
    def forward(self,x):
        f=self.features(x)
        context=[f]+[F.interpolate(F.avg_pool2d(f,s),size=f.shape[-2:],mode='bilinear',align_corners=True) for s in (2,4)]
        return self.fuse(torch.cat(context,1))

class RefinementNet(nn.Module):
    def __init__(self,variant='G3',channels=4):
        super().__init__();assert variant in ('G3','G4','G5')
        self.variant=variant;self.core=RestoredPansharpeningNet(channels,4)
        self.refiners=nn.ModuleList([Refiner(channels)])
        if variant=='G5':self.refiners.append(copy.deepcopy(self.refiners[0]))
        self.register_buffer('pan_weights',torch.ones(1,channels,1,1)/channels)
        self.register_buffer('pan_bias',torch.zeros(1))
        self.register_buffer('blur_kernel',torch.ones(channels,1,1,1))
    def configure(self,calibration):
        self.pan_weights.copy_(torch.tensor(calibration['pan_weights']).view_as(self.pan_weights));self.pan_bias.fill_(calibration['pan_bias'])
        k=torch.tensor(calibration['blur_kernel']);self.blur_kernel=k.view(1,1,*k.shape).repeat(self.pan_weights.shape[1],1,1,1).to(self.pan_weights.device)
    def down(self,x,size):
        p=self.blur_kernel.shape[-1]//2
        if p:x=F.conv2d(F.pad(x,(p,p,p,p),mode='reflect'),self.blur_kernel,groups=x.shape[1])
        return F.interpolate(x,size=size,mode='bilinear',align_corners=True)
    def gate_values(self):return {}
    def forward(self,pan,lms,ms):
        x=self.core(pan,lms,ms)
        for i in range(2):
            if self.variant=='G3':em=torch.zeros_like(x);ep=torch.zeros_like(pan)
            else:
                em=F.interpolate(ms-self.down(x,ms.shape[-2:]),size=x.shape[-2:],mode='bilinear',align_corners=True)
                ep=pan-(x*self.pan_weights).sum(1,keepdim=True)-self.pan_bias
                ep=ep-F.avg_pool2d(F.pad(ep,(2,2,2,2),mode='reflect'),5,stride=1)
            x=x+self.refiners[i if self.variant=='G5' else 0](torch.cat((x,lms,pan,em,ep),1))
        return x
