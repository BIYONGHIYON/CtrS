"""Experimental RGB-guided, latent spectral SSA-MRN (not the paper baseline)."""
import torch
from torch import nn
from torch.nn import functional as F

from .ssa_mrn import RestoredPansharpeningNet


class RGBLatentCore(RestoredPansharpeningNet):
    """Batch independent SSA branches using grouped convolutions; same parameters."""
    def __init__(self, channels, ssai_dimension, vectorized=True):
        super().__init__(channels, ssai_dimension, guide_channels=3)
        self.vectorized = vectorized

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        if not self.vectorized:
            return super()._attend(pan, features, ms, blocks, fusion, activation)
        groups = self.channels
        n, _, h, w = pan.shape
        # The guide is shared across branches: one convolution gives all branch outputs.
        guide = F.conv2d(pan, torch.cat([b.conv1t6.weight for b in blocks]),
                         torch.cat([b.conv1t6.bias for b in blocks]), padding=1).relu()
        guide = F.conv2d(guide, torch.cat([b.conv6t6.weight for b in blocks]),
                         torch.cat([b.conv6t6.bias for b in blocks]), padding=1, groups=groups)
        branch_input = torch.cat((features[:, None].expand(-1, groups, -1, -1, -1),
                                  ms[:, :, None]), dim=2).reshape(n, groups * (groups+1), h, w)
        projected = F.conv2d(branch_input, torch.cat([b.conv7t6_3.weight for b in blocks]),
                             torch.cat([b.conv7t6_3.bias for b in blocks]), padding=1, groups=groups)
        # Preserve the upstream width/height transpositions and per-channel spatial softmax.
        guide_flat = guide.reshape(n, -1, h*w)
        projected_flat = projected.transpose(-1, -2).reshape(n, -1, h*w)
        attention = (guide_flat * projected_flat).reshape(n, -1, h, w)
        attention = attention.transpose(-1, -2).reshape(n, -1, h*w).softmax(dim=-1)
        attended = (guide_flat * attention).reshape(n, -1, h, w)
        return activation(fusion(attended))


class RGBHSISsaMRN(nn.Module):
    def __init__(self, bands=204, latent_channels=8, ssai_dimension=6, vectorized=True):
        super().__init__()
        self.encoder = nn.Conv2d(bands, latent_channels, 1)
        self.core = RGBLatentCore(latent_channels, ssai_dimension, vectorized)
        self.decoder = nn.Conv2d(latent_channels, bands, 1)
        # Start near interpolation; every output band remains supervised.
        nn.init.normal_(self.decoder.weight, std=1e-3)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, rgb, lr_hsi):
        size = rgb.shape[-2:]
        latent_lr = self.encoder(lr_hsi)
        latent_hr = F.interpolate(latent_lr, size=size, mode="bilinear", align_corners=True)
        residual = self.core(rgb, latent_hr, latent_lr)
        baseline = F.interpolate(lr_hsi, size=size, mode="bicubic", align_corners=False)
        return baseline + self.decoder(residual)
