"""QB K4 structural ablations; preserve all shared upstream parameter initializations."""
import torch
from torch import nn
from .ssa_mrn import RestoredPansharpeningNet

VARIANTS = ('S0', 'S1', 'S2', 'S3', 'R1', 'R2', 'M1', 'I1', 'I2')

class StructureNet(RestoredPansharpeningNet):
    def __init__(self, variant='S0'):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(4, ssai_dimension=4)
        self.variant = variant
        # New layers are created after the unchanged upstream initialization.
        if variant == 'M1':
            self.ms_mix = nn.Sequential(nn.Conv2d(8, 16, 1), nn.PReLU(), nn.Conv2d(16, 4, 3, padding=1))
        elif variant == 'I1':
            old = self.cov2t64
            self.cov2t64 = nn.Conv2d(5, 64, 3, padding=1)
            with torch.no_grad():
                self.cov2t64.weight.copy_(old.weight[:, :5]); self.cov2t64.bias.copy_(old.bias)
        elif variant == 'I2':
            # 6->64 original:3520 parameters; split stem:3744 (+6.4%).
            self.pan_stem = nn.Sequential(nn.Conv2d(2, 20, 3, padding=1), nn.ReLU())
            self.ms_stem = nn.Sequential(nn.Conv2d(4, 20, 3, padding=1), nn.ReLU())
            self.stem_fuse = nn.Conv2d(40, 64, 1)
            del self.cov2t64

    def forward(self, pan, lms, ms):
        v = self.variant
        lowpan = self.prelu(self.conv1t1(self.upsample1(self.downsample1(pan))))
        if v == 'I1':
            f = self.cov2t64(torch.cat((lms, pan), 1))
        elif v == 'I2':
            f = self.stem_fuse(torch.cat((self.ms_stem(lms), self.pan_stem(torch.cat((pan, lowpan), 1))), 1))
        else:
            f = self.cov2t64(torch.cat((lms, pan, lowpan), 1))
        res = self.conv64t48(self.backbone(f)) + lms
        if v != 'S1':
            res = res + self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        half = self.downsample100(res)
        if v != 'S2':
            half = half + self._attend(self.downsample100(pan), half, self.upsample100(ms), self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        if v == 'S3':
            small = torch.zeros_like(ms)
        elif v == 'M1':
            small = self.ms_mix(torch.cat((self.downsample200(half), ms), 1))
        else:
            small = self._attend(self.downsample2(pan), self.downsample200(half), ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)
        out = self.conv2ctc1(torch.cat((ms, small), 1))
        if v == 'R2': out = ms + out
        up = self.upsample101(out)
        out = self.conv2ctc2(torch.cat((up, half), 1))
        if v == 'R2': out = up + out
        up = self.upsample102(out)
        out = self.conv2ctc3(torch.cat((up, res), 1))
        if v == 'R2': out = up + out
        if v == 'R1': out = lms + out
        return out
