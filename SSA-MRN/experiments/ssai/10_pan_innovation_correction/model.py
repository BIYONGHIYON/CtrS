"""SSAI-only signed detail correction; fixed train-input PAN response."""
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

STAGES = ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2')


def lowpass(x):
    return F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), 5, stride=1)


def highpass(x):
    return x - lowpass(x)


class InnovationSSA(nn.Module):
    def __init__(self, source, innovation):
        super().__init__()
        self.fixed = source.fixed
        self.innovation = innovation
        # Preserve every original layer and common initialization, including
        # forward-unused conv6t12. Registered and active counts are reported apart.
        for name, child in source.named_children():
            self.add_module(name, child)
        self.register_buffer('pan_response', torch.tensor([.25, .25, .25, .25, 0.]))
        self.gamma = nn.Parameter(torch.tensor(.1))

    def forward(self, pan, band_input):
        predicted_pan = (band_input[:, :4] * self.pan_response[:4].view(1, 4, 1, 1)).sum(1, keepdim=True) + self.pan_response[4]
        error = pan - predicted_pan if self.innovation else pan
        detail = highpass(error)
        # Bias-free evaluation makes zero innovation give exactly zero detail.
        # Original bias tensors are retained for source/state compatibility.
        p = F.conv2d(self.relu(F.conv2d(detail, self.conv1t6.weight, padding=1)), self.conv6t6.weight, padding=1)
        gate = 2 * self.conv7t6_3(band_input).sigmoid()
        # Sign survives the final unconstrained convolution and highpass;
        # no nonnegative output, spatial softmax, area division or RMS rescaling.
        return self.gamma * highpass(p * gate)


class PanInnovationNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4, innovation=True):
        if channels != 4 or ssai_dimension != 4:
            raise ValueError('QB C4 / K4 control only')
        super().__init__(channels, ssai_dimension)
        for stage in STAGES:
            blocks = getattr(self, stage)
            for i, source in enumerate(blocks):
                blocks[i] = InnovationSSA(source, innovation)

    def set_response(self, coefficients):
        with torch.no_grad():
            for stage in STAGES:
                for block in getattr(self, stage):
                    block.pan_response.copy_(torch.as_tensor(coefficients, dtype=block.pan_response.dtype, device=block.pan_response.device))
