"""QB K4: coarse feature bypass and adjustable PAN contributions."""
import torch
from torch import nn
StructureNet = __import__('importlib').import_module('research.performance.17_structural_ablation.model').StructureNet
VARIANTS = ('N0', 'N1', 'N2', 'N3', 'N4', 'N5')

class CoarseAnchorNet(StructureNet):
    def __init__(self, variant='N0'):
        if variant not in VARIANTS: raise ValueError(variant)
        super().__init__({'N0':'S0','N1':'I1','N2':'M1','N3':'M1','N4':'S0','N5':'M1'}[variant])
        self.experiment = variant
        if variant in ('N3','N5'): self.coarse_gain = nn.Parameter(torch.zeros(()))
        if variant in ('N4','N5'): self.lrpan_gain = nn.Parameter(torch.ones(()))

    def gate_values(self):
        return {k:float(getattr(self,k).detach().cpu()) for k in ('coarse_gain','lrpan_gain') if hasattr(self,k)}

    def forward(self, pan, lms, ms):
        if self.experiment in ('N0','N1','N2'): return super().forward(pan,lms,ms)
        lowpan=self.prelu(self.conv1t1(self.upsample1(self.downsample1(pan))))
        if hasattr(self,'lrpan_gain'): lowpan=lowpan*self.lrpan_gain
        f=self.cov2t64(torch.cat((lms,pan,lowpan),1))
        res=self.conv64t48(self.backbone(f))+lms
        res=res+self._attend(pan,res,lms,self.SSA_blocks,self.conv48tnum1,self.prelu1)
        half=self.downsample100(res)
        half=half+self._attend(self.downsample100(pan),half,self.upsample100(ms),self.SSA_blocks1,self.conv48tnum2,self.prelu2)
        quarter=self.downsample200(half)
        if hasattr(self,'coarse_gain'):
            mixed=self.ms_mix(torch.cat((quarter,ms),1))
            attention=self._attend(self.downsample2(pan),quarter,ms,self.SSA_blocks2,self.conv48tnum3,self.prelu3)
            small=mixed+self.coarse_gain*attention
        else:
            small=self._attend(self.downsample2(pan),quarter,ms,self.SSA_blocks2,self.conv48tnum3,self.prelu3)
        out=self.conv2ctc1(torch.cat((ms,small),1))
        out=self.conv2ctc2(torch.cat((self.upsample101(out),half),1))
        return self.conv2ctc3(torch.cat((self.upsample102(out),res),1))
