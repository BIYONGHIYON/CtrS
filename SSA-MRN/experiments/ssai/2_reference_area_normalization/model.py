"""Inference diagnostic: keep the mean gate at its training-reference scale."""
from torch import nn
from common.model import RestoredPansharpeningNet

class ReferenceAreaSSA(nn.Module):
    def __init__(self, source, reference_area):
        super().__init__()
        self.fixed = source.fixed
        self.reference_area = reference_area
        self._original_forward = type(source).forward
        for name, child in source.named_children():
            self.add_module(name, child)

    def forward(self, pan, ms_pro):
        output = self._original_forward(self, pan, ms_pro)
        area = output.shape[-2] * output.shape[-1]
        if area == self.reference_area:
            return output
        return output * (area / self.reference_area)

class ReferenceAreaNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4):
        if ssai_dimension != 4:
            raise ValueError("This experiment fixes K=4")
        super().__init__(channels, ssai_dimension)
        for name, reference_area in zip(
            ("SSA_blocks", "SSA_blocks1", "SSA_blocks2"), (4096, 1024, 256)
        ):
            blocks = getattr(self, name)
            for band, source in enumerate(blocks):
                blocks[band] = ReferenceAreaSSA(source, reference_area)

    def forward(self, pan, lms, ms):
        if self.training and tuple(lms.shape[-2:]) != (64, 64):
            raise ValueError("Training-reference geometry must remain 64x64")
        return super().forward(pan, lms, ms)
