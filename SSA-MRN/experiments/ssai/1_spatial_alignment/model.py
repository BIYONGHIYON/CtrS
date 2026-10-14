"""SSAI coordinate alignment only; all original parameter names are retained."""
import torch
from torch import nn
from common.model import RestoredPansharpeningNet

class AlignedSSA(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.fixed = source.fixed
        for name, child in source.named_children():
            self.add_module(name, child)

    def forward(self, pan, ms_pro):
        p = self.conv6t6(self.relu(self.conv1t6(pan)))
        q = self.conv7t6_3(ms_pro)
        weights = (p * q).flatten(2).softmax(-1).reshape_as(p)
        return p * weights

class SpatialAlignmentNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4):
        if ssai_dimension != 4:
            raise ValueError("This experiment fixes K=4")
        super().__init__(channels, ssai_dimension)
        for name in ("SSA_blocks", "SSA_blocks1", "SSA_blocks2"):
            blocks = getattr(self, name)
            for band, source in enumerate(blocks):
                blocks[band] = AlignedSSA(source)
