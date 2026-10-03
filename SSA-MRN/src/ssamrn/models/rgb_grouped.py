"""Grouped spectral SSA-MRN with scalar RGB guides and 23-tap interpolation."""
import torch
from torch import nn
from torch.nn import functional as F
from .rgb_hsi import RGBLatentCore, RGBHSISsaMRN
from .interp23 import interp23tap, Interp23


def _set_core_resampling(core, mode):
    if mode == 'bilinear':
        # Match every resize used by the PAN-MS reproduction's forward path.
        for name, scale in [('upsample1',4),('upsample100',2),('upsample101',2),('upsample102',2),
                            ('downsample1',.25),('downsample2',.25),('downsample100',.5),('downsample200',.5)]:
            setattr(core, name, nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
    elif mode != 'area_23tap':
        raise ValueError('Unknown core resampling: '+mode)


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


class ScalarGrouped12Core(RGBGroupedCore):
    """One scalar guide drives all twelve spectral branches at every scale."""
    def __init__(self, core_resampling='area_23tap'):
        super().__init__()
        self.guide_channels = 1
        self.conv1t1 = nn.Conv2d(1, 1, 3, padding=1)
        self.cov2t64 = nn.Conv2d(14, 64, 3, padding=1)
        _set_core_resampling(self, core_resampling)

    def _attend(self, guide, features, ms, blocks, fusion, activation):
        # Reuse the vectorized implementation: all three guide groups see
        # the same scalar input, while their branch weights remain independent.
        return super()._attend(guide.expand(-1, 3, -1, -1), features, ms,
                               blocks, fusion, activation)


class ScalarGrouped34Core(RGBLatentCore):
    """One scalar RGB guide drives 34 branches; each feature represents six bands."""
    def __init__(self, core_resampling='area_23tap'):
        super().__init__(34, 4, vectorized=True)
        self.guide_channels = 1
        self.conv1t1 = nn.Conv2d(1, 1, 3, padding=1)
        self.cov2t64 = nn.Conv2d(36, 64, 3, padding=1)
        for blocks in (self.SSA_blocks, self.SSA_blocks1, self.SSA_blocks2):
            for block in blocks:
                block.conv1t6 = nn.Conv2d(1, 4, 3, padding=1)
        for name, ratio in [('upsample1',4),('upsample100',2),('upsample101',2),('upsample102',2)]:
            setattr(self, name, Interp23(ratio))
        for name, ratio in [('downsample1',4),('downsample2',4),('downsample100',2),('downsample200',2)]:
            setattr(self, name, nn.AvgPool2d(ratio))
        _set_core_resampling(self, core_resampling)

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        n, _, h, w = pan.shape
        guide = F.conv2d(pan, torch.cat([b.conv1t6.weight for b in blocks]),
                         torch.cat([b.conv1t6.bias for b in blocks]), padding=1).relu()
        guide = F.conv2d(guide, torch.cat([b.conv6t6.weight for b in blocks]),
                         torch.cat([b.conv6t6.bias for b in blocks]), padding=1, groups=34)
        # Split the linear projection into shared features + one per-branch band.
        # This equals conv(cat(features, band)) without a 34*35-channel HR tensor.
        shared_weights = torch.cat([b.conv7t6_3.weight[:, :34] for b in blocks])
        band_weights = torch.cat([b.conv7t6_3.weight[:, 34:] for b in blocks])
        projected = F.conv2d(features, shared_weights, padding=1)
        projected = projected + F.conv2d(ms, band_weights,
                         torch.cat([b.conv7t6_3.bias for b in blocks]), padding=1, groups=34)
        guide_flat = guide.reshape(n, 136, h*w)
        attention = (guide_flat * projected.transpose(-1, -2).reshape(n, 136, h*w)).reshape(n, 136, h, w)
        attention = attention.transpose(-1, -2).reshape(n, 136, h*w).softmax(dim=-1)
        return activation(fusion((guide_flat * attention).reshape(n, 136, h, w)))


class RGBTripleGroupedSsaMRN(nn.Module):
    """R/G/B independently reconstruct all 204 bands; learn bandwise fusion."""
    def __init__(self, latent_channels=12, core_resampling='area_23tap'):
        super().__init__()
        if latent_channels not in (12, 34):
            raise ValueError('Triple grouped model supports 12 or 34 features')
        self.encoder = nn.Conv2d(204, latent_channels, 1, groups=latent_channels)
        if latent_channels == 12:
            self.cores = nn.ModuleList([ScalarGrouped12Core(core_resampling) for _ in range(3)])
        else:
            self.cores = nn.ModuleList([ScalarGrouped34Core(core_resampling) for _ in range(3)])
        self.decoders = nn.ModuleList([nn.Conv2d(latent_channels, 204, 1, groups=latent_channels) for _ in range(3)])
        self.fusion = nn.Conv2d(204 * 3, 204, 1, groups=204)
        for decoder in self.decoders:
            nn.init.normal_(decoder.weight, std=1e-3)
            nn.init.zeros_(decoder.bias)
        nn.init.constant_(self.fusion.weight, 1 / 3)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, rgb, lr_hsi):
        if rgb.ndim != 4 or lr_hsi.ndim != 4:
            raise ValueError('Expected NCHW RGB and HSI')
        if rgb.shape[1] != 3 or lr_hsi.shape[1] != 204 or rgb.shape[0] != lr_hsi.shape[0]:
            raise ValueError('Expected matching batches, RGB=3 and HSI=204')
        if rgb.shape[-2:] != tuple(s * 4 for s in lr_hsi.shape[-2:]):
            raise ValueError('Expected x4 spatial ratio')
        latent = self.encoder(lr_hsi)
        up = interp23tap(latent, 4)
        corrections = [decoder(core(rgb[:, i:i+1], up, latent))
                       for i, (core, decoder) in enumerate(zip(self.cores, self.decoders))]
        # Interleave R/G/B estimates for each output band before grouped fusion.
        n, c, h, w = corrections[0].shape
        per_band = torch.stack(corrections, dim=2).reshape(n, c * 3, h, w)
        return interp23tap(lr_hsi, 4) + self.fusion(per_band)


def build_rgb_hsi_model(config):
    kind=config.get('model_type','latent_rgb')
    if kind in ('rgb_grouped12_23tap', 'rgb_triple_grouped12_23tap', 'rgb_triple_grouped34_23tap', 'rgb_triple_grouped34_bilinear', 'rgb_triple_grouped12_bilinear'):
        channels = 34 if kind in ('rgb_triple_grouped34_23tap', 'rgb_triple_grouped34_bilinear') else 12
        if config['latent_channels'] != channels or config['ssai_dimension'] != 4:
            raise ValueError(f'Grouped model requires {channels} features and K=4')
        if config.get('upsampler') != '23tap':
            raise ValueError('Grouped model requires upsampler=23tap')
        mode = 'bilinear' if kind.endswith('_bilinear') else 'area_23tap'
        return RGBGroupedSsaMRN() if kind == 'rgb_grouped12_23tap' else RGBTripleGroupedSsaMRN(channels, mode)
    if kind!='latent_rgb':
        raise ValueError('Unknown model_type: '+kind)
    return RGBHSISsaMRN(204,config['latent_channels'],config['ssai_dimension'],config.get('vectorized_ssa',True))
