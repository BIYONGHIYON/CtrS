"""Spatial memory selection, post-restoration observation, high-pass PAN guidance."""
import torch
from torch import nn
from torch.nn import functional as F
from .recurrent_guidance import GuidanceNet
VARIANTS=('E0','E1','E5','E3','E6','E4','E7','E2')
PARENTS={'E0':'D0','E1':'D3','E2':'D4','E3':'D5','E4':'D6','E5':'D3','E6':'D5','E7':'D6'}
class RefinementNet(GuidanceNet):
    def __init__(self,variant='E0'):
        if variant not in PARENTS:raise ValueError(variant)
        super().__init__(PARENTS[variant])
        self.arm=variant
        if variant=='E5':
            self.spatial_gate=nn.ModuleList([nn.Conv2d(8,4,1) for _ in range(3)])
            for layer in self.spatial_gate:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    @staticmethod
    def highpass(x):
        return x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,stride=1)
    def _memory(self,feature,previous,scale):
        if self.arm!='E5':return super()._memory(feature,previous,scale)
        if previous is None:return feature
        logits=self.memory_gate[scale].view(1,4,1,1)+self.spatial_gate[scale](torch.cat((feature,previous[scale]),1))
        return feature+torch.tanh(logits)*self.memory_projection[scale](previous[scale])
    def _attend(self,pan,features,ms,scale):
        if self.arm!='E7':return super()._attend(pan,features,ms,scale)
        core=self.core
        blocks=(core.SSA_blocks,core.SSA_blocks1,core.SSA_blocks2)[scale]
        fusion=(core.conv48tnum1,core.conv48tnum2,core.conv48tnum3)[scale]
        activation=(core.prelu1,core.prelu2,core.prelu3)[scale]
        outputs=[]
        for band,block in enumerate(blocks):
            band_input=torch.cat((features,ms[:,band:band+1]),1)
            attended=block(pan,band_input)
            pan_feature=block.conv6t6(block.relu(block.conv1t6(pan)))
            ms_feature=block.conv7t6_3(band_input)
            difference=self.highpass(pan_feature-self.pan_projection[scale][band](ms_feature))
            outputs.append(attended+torch.tanh(self.pan_gate[scale,band])*difference)
        return activation(fusion(torch.cat(outputs,1)))
    def forward(self,pan,lms,ms):
        if self.arm!='E6':return super().forward(pan,lms,ms)
        if lms.shape[1]!=4 or ms.shape!=(len(lms),4,lms.shape[-2]//4,lms.shape[-1]//4):raise ValueError('QB scale4 inputs required')
        current=lms;previous=None
        for _ in range(3):
            zero=torch.zeros_like(current)
            proposal,previous=self._restore(pan,current+self.error_encoder(zero),ms,previous)
            restored=current+self.alpha*(proposal-current)
            observed=F.interpolate(restored,size=ms.shape[-2:],mode='bilinear',align_corners=True)
            error=F.interpolate(ms-observed,size=restored.shape[-2:],mode='bilinear',align_corners=True)
            error=torch.tanh(self.observation_gate).view(1,4,1,1)*error
            correction=self.error_encoder(error)-self.error_encoder(zero)
            current=restored+self.alpha*correction
        return current
