"""Shared MS-anchored affine transform; no learned parameters or GT statistics."""
import torch
from torch import nn
from common.model import RestoredPansharpeningNet

class Model(nn.Module):
    def __init__(self, arm, std_floor=0.01):
        super().__init__()
        if arm not in ('N0', 'N1') or std_floor <= 0:
            raise ValueError('Invalid arm or standard deviation floor')
        self.core = RestoredPansharpeningNet(4, ssai_dimension=4)
        self.arm = arm
        self.std_floor = float(std_floor)

    def statistics(self, ms):
        # Population variance, jointly across bands and spatial pixels, per sample.
        # Explicit FP32 statistics under FP16 autocast; never pool batch dimension.
        with torch.autocast(device_type=ms.device.type, enabled=False):
            x = ms.float()
            var, mean = torch.var_mean(x, dim=(1, 2, 3), correction=0, keepdim=True)
            scale = var.clamp_min(0).sqrt().clamp_min(self.std_floor)
        return mean, scale

    def forward(self, pan, lms, ms):
        if self.arm == 'N0':
            return self.core(pan, lms, ms)
        mean, scale = self.statistics(ms)
        # The same one affine transform for PAN, LMS and all four MS bands.
        with torch.autocast(device_type=ms.device.type, enabled=False):
            inputs = tuple((x.float() - mean) / scale for x in (pan, lms, ms))
        prediction = self.core(*inputs)
        with torch.autocast(device_type=ms.device.type, enabled=False):
            return prediction.float() * scale + mean
