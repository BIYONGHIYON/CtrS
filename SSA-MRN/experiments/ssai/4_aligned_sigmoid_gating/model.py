"""Same-coordinate PAN/MS product with the fixed-reference sigmoid gate."""
from torch import nn
from common.model import RestoredPansharpeningNet

class AlignedSigmoidSSA(nn.Module):
    def __init__(self, source, reference_area):
        super().__init__()
        self.fixed = source.fixed
        self.reference_area = reference_area
        for name, child in source.named_children():
            self.add_module(name, child)

    def forward(self, pan, ms_pro):
        p = self.conv6t6(self.relu(self.conv1t6(pan)))
        q = self.conv7t6_3(ms_pro)
        return p * ((2.0 / self.reference_area) * (p * q).sigmoid())

class AlignedSigmoidNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4):
        if ssai_dimension != 4:
            raise ValueError("This experiment fixes K=4")
        super().__init__(channels, ssai_dimension)
        for name, area in zip(("SSA_blocks", "SSA_blocks1", "SSA_blocks2"), (4096, 1024, 256)):
            blocks = getattr(self, name)
            for band, source in enumerate(blocks):
                blocks[band] = AlignedSigmoidSSA(source, area)

    def forward(self, pan, lms, ms):
        if self.training and tuple(lms.shape[-2:]) != (64, 64):
            raise ValueError("Training-reference geometry must remain 64x64")
        return super().forward(pan, lms, ms)
