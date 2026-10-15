"""Keep original coordinates and parameters; normalize spatial weight mean to one."""
import torch
from torch import nn
from common.model import RestoredPansharpeningNet


class MeanNormalizedSSA(nn.Module):
    def __init__(self, source, learn_gamma=False):
        super().__init__()
        self.fixed = source.fixed
        for name, child in source.named_children():
            self.add_module(name, child)
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.gamma = 1.0
        self.normalize_mean = True

    def components(self, pan, ms_pro):
        p = self.conv6t6(self.relu(self.conv1t6(pan)))
        q = self.conv7t6_3(ms_pro)
        b, c, h, w = q.shape
        # Identical two transposes and rectangular reshapes to experiment3.
        z = p.reshape(b, c, -1) * q.transpose(-1, -2).reshape(b, c, -1)
        z = z.reshape(b, c, h, w).transpose(-1, -2).reshape(b, c, -1)
        weight = z.softmax(dim=-1)
        if self.normalize_mean:
            weight = weight * (h * w)
        return p, weight.reshape(b, c, h, w)

    def forward(self, pan, ms_pro):
        p, weight = self.components(pan, ms_pro)
        return p * weight * self.gamma


class MeanNormalizedNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4, learn_gamma=False):
        if channels != 4 or ssai_dimension != 4:
            raise ValueError('QB C4 and output K4 are fixed')
        super().__init__(channels, ssai_dimension)
        for name in ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2'):
            blocks = getattr(self, name)
            for band, source in enumerate(blocks):
                blocks[band] = MeanNormalizedSSA(source, learn_gamma)
