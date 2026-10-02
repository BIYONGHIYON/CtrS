"""12 spectral groups: four SSA branches per scalar R/G/B guide, K=4."""
import torch
from torch import nn
from torch.nn import functional as F
from .rgb_hsi import RGBLatentCore, RGBHSISsaMRN
from .interp23 import interp23tap, Interp23


class RGBGroupedCore(RGBLatentCore):
    def __init__(self):
        super().__init__(12,4,vectorized=True)
        for blocks in (self.SSA_blocks,self.SSA_blocks1,self.SSA_blocks2):
            for block in blocks:
                block.conv1t6=nn.Conv2d(1,4,3,padding=1)
        # Replace every resize actually used by the restored forward path.
        for name,ratio in [('upsample1',4),('upsample100',2),('upsample101',2),('upsample102',2)]:
            setattr(self,name,Interp23(ratio))
        for name,ratio in [('downsample1',4),('downsample2',4),('downsample100',2),('downsample200',2)]:
            setattr(self,name,nn.AvgPool2d(ratio))

    def _attend(self,pan,features,ms,blocks,fusion,activation):
        n,_,h,w=pan.shape
        # Outputs 0..15 use R only, 16..31 G only, 32..47 B only.
        guide=F.conv2d(pan,torch.cat([b.conv1t6.weight for b in blocks]),
                       torch.cat([b.conv1t6.bias for b in blocks]),padding=1,groups=3).relu()
        guide=F.conv2d(guide,torch.cat([b.conv6t6.weight for b in blocks]),
                       torch.cat([b.conv6t6.bias for b in blocks]),padding=1,groups=12)
        inputs=torch.cat([features[:,None].expand(-1,12,-1,-1,-1),ms[:,:,None]],dim=2).reshape(n,156,h,w)
        projected=F.conv2d(inputs,torch.cat([b.conv7t6_3.weight for b in blocks]),
                           torch.cat([b.conv7t6_3.bias for b in blocks]),padding=1,groups=12)
        g=guide.reshape(n,48,h*w)
        attention=(g*projected.transpose(-1,-2).reshape(n,48,h*w)).reshape(n,48,h,w)
        attention=attention.transpose(-1,-2).reshape(n,48,h*w).softmax(-1)
        return activation(fusion((g*attention).reshape(n,48,h,w)))


class RGBGroupedSsaMRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Conv2d(204,12,1,groups=12)
        self.core=RGBGroupedCore()
        self.decoder=nn.Conv2d(12,204,1,groups=12)
        nn.init.normal_(self.decoder.weight,std=1e-3);nn.init.zeros_(self.decoder.bias)

    def forward(self,rgb,lr_hsi):
        if rgb.shape[1]!=3 or lr_hsi.shape[1]!=204 or rgb.shape[-2:]!=tuple(s*4 for s in lr_hsi.shape[-2:]):
            raise ValueError('Expected RGB HR and 204-band HSI at x4 spatial ratio')
        latent=self.encoder(lr_hsi)
        residual=self.core(rgb,interp23tap(latent,4),latent)
        return interp23tap(lr_hsi,4)+self.decoder(residual)


def build_rgb_hsi_model(config):
    kind=config.get('model_type','latent_rgb')
    if kind=='rgb_grouped12_23tap':
        if config['latent_channels']!=12 or config['ssai_dimension']!=4:
            raise ValueError('Grouped model requires 12 features and K=4')
        if config.get('upsampler') != '23tap':
            raise ValueError('Grouped model requires upsampler=23tap')
        return RGBGroupedSsaMRN()
    if kind!='latent_rgb':
        raise ValueError('Unknown model_type: '+kind)
    return RGBHSISsaMRN(204,config['latent_channels'],config['ssai_dimension'],config.get('vectorized_ssa',True))
