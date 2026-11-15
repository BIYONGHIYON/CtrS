"""Shared three-stage restoration with stage-specific updates and feature conditioning."""
import importlib
import torch
from torch import nn
from torch.nn import functional as F
GuidanceNet=importlib.import_module('research.performance.20_recurrent_guidance.model').GuidanceNet
VARIANTS=('F0','F1','F2','F3','F4','F5')
class StageAwareNet(GuidanceNet):
    def __init__(self,variant='F0'):
        if variant not in VARIANTS:raise ValueError(variant)
        super().__init__('D0' if variant=='F5' else 'D3')
        self.arm=variant
        self._stage=0
        if variant in ('F1','F3','F4'):
            self.stage_alpha=nn.Parameter(torch.full((3,),0.1))
            self.alpha.requires_grad_(False)
        if variant in ('F2','F3','F4'):
            # Zero-initialized affine code per restoration stage, spatial scale and band.
            self.stage_scale=nn.Parameter(torch.zeros(3,3,4))
            self.stage_bias=nn.Parameter(torch.zeros(3,3,4))
        if variant=='F4':
            self.pan_projection=nn.ModuleList([nn.ModuleList([nn.Conv2d(4,4,3,padding=1) for _ in range(4)]) for _ in range(3)])
            self.pan_gate=nn.Parameter(torch.zeros(3,4))
    def _memory(self,feature,previous,scale):
        feature=super()._memory(feature,previous,scale)
        if hasattr(self,'stage_scale'):
            feature=feature*(1+self.stage_scale[self._stage,scale].view(1,4,1,1))+self.stage_bias[self._stage,scale].view(1,4,1,1)
        return feature
    def _attend(self,pan,features,ms,scale):
        if self.arm!='F4' or self._stage!=2:
            return super()._attend(pan,features,ms,scale)
        core=self.core
        blocks=(core.SSA_blocks,core.SSA_blocks1,core.SSA_blocks2)[scale]
        fusion=(core.conv48tnum1,core.conv48tnum2,core.conv48tnum3)[scale]
        activation=(core.prelu1,core.prelu2,core.prelu3)[scale]
        outputs=[]
        for band,block in enumerate(blocks):
            band_input=torch.cat((features,ms[:,band:band+1]),1)
            attended=block(pan,band_input)
            pf=block.conv6t6(block.relu(block.conv1t6(pan)))
            mf=block.conv7t6_3(band_input)
            difference=pf-self.pan_projection[scale][band](mf)
            hp=difference-F.avg_pool2d(F.pad(difference,(2,2,2,2),mode='reflect'),5,stride=1)
            outputs.append(attended+torch.tanh(self.pan_gate[scale,band])*hp)
        return activation(fusion(torch.cat(outputs,1)))
    def _forward(self,pan,lms,ms,trace=False):
        if self.arm=='F5':
            result=self.core(pan,lms,ms)
            return [result] if trace else result
        if pan.shape!=(len(lms),1,*lms.shape[-2:]) or lms.shape[1]!=4 or ms.shape!=(len(lms),4,lms.shape[-2]//4,lms.shape[-1]//4):raise ValueError('QB NCHW scale4 inputs required')
        if lms.shape[-2]%4 or lms.shape[-1]%4:raise ValueError('LMS must be divisible by4')
        current=lms;previous=None;outputs=[]
        try:
            for stage in range(3):
                self._stage=stage
                corrected=current+self.error_encoder(torch.zeros_like(current))
                proposal,previous=self._restore(pan,corrected,ms,previous)
                alpha=self.stage_alpha[stage] if hasattr(self,'stage_alpha') else self.alpha
                current=current+alpha*(proposal-current)
                if trace:outputs.append(current)
        finally:self._stage=0
        return outputs if trace else current
    def forward(self,pan,lms,ms):return self._forward(pan,lms,ms)
    def forward_stages(self,pan,lms,ms):return self._forward(pan,lms,ms,True)
    def gate_values(self):
        result=super().gate_values()
        if hasattr(self,'stage_alpha'):
            for i,x in enumerate(self.stage_alpha):result[f'alpha_stage_{i+1}']=float(x.detach().cpu())
        for name in ('stage_scale','stage_bias'):
            if hasattr(self,name):result[name+'_abs_mean']=float(getattr(self,name).detach().abs().mean().cpu())
        return result
