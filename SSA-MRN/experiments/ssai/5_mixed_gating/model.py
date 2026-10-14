"""Mix original spatial softmax and fixed-reference sigmoid without new parameters."""
from torch import nn
from common.model import RestoredPansharpeningNet

class MixedSSA(nn.Module):
    def __init__(self, source, reference_area, alpha):
        super().__init__()
        self.fixed = source.fixed
        self.reference_area = reference_area
        self.alpha = alpha
        for name, child in source.named_children():
            self.add_module(name, child)

    def forward(self, pan, ms_pro):
        p = self.conv6t6(self.relu(self.conv1t6(pan)))
        q = self.conv7t6_3(ms_pro)
        b, c, h, w = q.shape
        z = p.reshape(b,c,-1) * q.transpose(-1,-2).reshape(b,c,-1)
        z = z.reshape(b,c,h,w).transpose(-1,-2).reshape(b,c,-1)
        gate = (1-self.alpha)*z.softmax(dim=-1) + self.alpha*(2/self.reference_area)*z.sigmoid()
        return (p.reshape(b,c,-1)*gate).reshape(b,c,h,w)

class MixedGatingNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=4, alpha=0.25):
        if ssai_dimension != 4 or not 0 <= alpha <= 1:
            raise ValueError("K=4, alpha in [0,1] required")
        super().__init__(channels, ssai_dimension)
        for name, area in zip(("SSA_blocks","SSA_blocks1","SSA_blocks2"),(4096,1024,256)):
            blocks = getattr(self,name)
            for band, source in enumerate(blocks):
                blocks[band] = MixedSSA(source,area,alpha)

    def forward(self,pan,lms,ms):
        if self.training and tuple(lms.shape[-2:]) != (64,64):
            raise ValueError("Training reference must remain 64x64")
        return super().forward(pan,lms,ms)
