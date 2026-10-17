"""Experiment40: 2x2 ablation of shared MS centering and scaling."""
import torch
from torch import nn
from common.model import RestoredPansharpeningNet

ARMS = ('A0', 'A1', 'A2', 'A3')
class Model(nn.Module):
    def __init__(self, arm, std_floor=0.01):
        super().__init__()
        if arm not in ARMS or std_floor <= 0:
            raise ValueError('Invalid arm/std floor')
        self.core = RestoredPansharpeningNet(4, ssai_dimension=4)
        self.arm = arm
        self.std_floor = float(std_floor)

    def statistics(self, ms):
        with torch.autocast(device_type=ms.device.type, enabled=False):
            variance, mean = torch.var_mean(ms.float(), dim=(1,2,3), correction=0, keepdim=True)
            scale = variance.clamp_min(0).sqrt().clamp_min(self.std_floor)
        return mean, scale

    def transform_parameters(self, ms):
        mean, scale = self.statistics(ms)
        return (mean if self.arm in ('A1','A3') else torch.zeros_like(mean),
                scale if self.arm in ('A2','A3') else torch.ones_like(scale))

    def forward(self, pan, lms, ms):
        if self.arm == 'A0':
            return self.core(pan, lms, ms)
        mean, scale = self.transform_parameters(ms)
        with torch.autocast(device_type=ms.device.type, enabled=False):
            inputs = tuple((x.float()-mean)/scale for x in (pan,lms,ms))
        prediction = self.core(*inputs)
        with torch.autocast(device_type=ms.device.type, enabled=False):
            return prediction.float()*scale+mean
