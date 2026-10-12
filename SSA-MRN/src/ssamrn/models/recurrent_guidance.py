"""K4 SSA-MRN shared restoration with cross-stage memory and observation/PAN differences."""
import torch
from torch import nn
from torch.nn import functional as F
from .ssa_mrn import RestoredPansharpeningNet

VARIANTS = ('D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6')


class GuidanceNet(nn.Module):
    """Original per-band SSAI at all three scales; new paths start with zero gates.

    Memory stays inside one forward, is differentiable across restoration stages,
    and is reset for each batch. The fixed observation is bilinear scale 4 with
    align_corners=True, an operational approximation rather than calibrated MTF.
    """
    def __init__(self, variant='D0'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.core = RestoredPansharpeningNet(4, 4)
        if variant == 'D0':
            return
        # Construct these before optional modules: all repeated controls share initialization.
        self.error_encoder = nn.Conv2d(4, 4, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.use_memory = variant in ('D3', 'D5', 'D6')
        self.use_observation = variant in ('D4', 'D5')
        self.use_pan_difference = variant == 'D6'
        if self.use_memory:
            self.memory_projection = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3)])
            self.memory_gate = nn.Parameter(torch.zeros(3, 4))
        if self.use_observation:
            self.observation_gate = nn.Parameter(torch.zeros(4))
        if self.use_pan_difference:
            self.pan_projection = nn.ModuleList([
                nn.ModuleList([nn.Conv2d(4, 4, 3, padding=1) for _ in range(4)])
                for _ in range(3)
            ])
            self.pan_gate = nn.Parameter(torch.zeros(3, 4))

    def gate_values(self):
        if self.variant == 'D0':
            return {}
        values = {'alpha': float(self.alpha.detach().cpu())}
        for name in ('memory_gate', 'observation_gate', 'pan_gate'):
            if hasattr(self, name):
                gate = torch.tanh(getattr(self, name).detach()).cpu()
                values[name + '_mean'] = float(gate.mean())
                values[name + '_abs_mean'] = float(gate.abs().mean())
                values[name + '_max_abs'] = float(gate.abs().max())
        return values

    def _memory(self, feature, previous, scale):
        if not self.use_memory or previous is None:
            return feature
        gate = torch.tanh(self.memory_gate[scale]).view(1, 4, 1, 1)
        return feature + gate * self.memory_projection[scale](previous[scale])

    def _attend(self, pan, features, ms, scale):
        core = self.core
        blocks = (core.SSA_blocks, core.SSA_blocks1, core.SSA_blocks2)[scale]
        fusion = (core.conv48tnum1, core.conv48tnum2, core.conv48tnum3)[scale]
        activation = (core.prelu1, core.prelu2, core.prelu3)[scale]
        outputs = []
        for band, block in enumerate(blocks):
            band_input = torch.cat((features, ms[:, band:band + 1]), dim=1)
            attended = block(pan, band_input)
            if self.use_pan_difference:
                # Aligned K4 feature difference; preserve the official SSA transpose path.
                pan_feature = block.conv6t6(block.relu(block.conv1t6(pan)))
                ms_feature = block.conv7t6_3(band_input)
                difference = pan_feature - self.pan_projection[scale][band](ms_feature)
                attended = attended + torch.tanh(self.pan_gate[scale, band]) * difference
            outputs.append(attended)
        return activation(fusion(torch.cat(outputs, dim=1)))

    def _restore(self, pan, lms, ms, previous):
        core = self.core
        pan_updown = core.prelu(core.conv1t1(core.upsample1(core.downsample1(pan))))
        features = core.backbone(core.cov2t64(torch.cat((lms, pan, pan_updown), dim=1)))
        res = core.conv64t48(features) + lms
        res = res + self._attend(pan, res, lms, 0)
        res = self._memory(res, previous, 0)
        half = core.downsample100(res)
        half = half + self._attend(core.downsample100(pan), half, core.upsample100(ms), 1)
        half = self._memory(half, previous, 1)
        quarter = core.downsample200(half)
        attended = self._attend(core.downsample2(pan), quarter, ms, 2)
        attended = self._memory(attended, previous, 2)
        output = core.conv2ctc1(torch.cat((ms, attended), dim=1))
        output = core.conv2ctc2(torch.cat((core.upsample101(output), half), dim=1))
        output = core.conv2ctc3(torch.cat((core.upsample102(output), res), dim=1))
        return output, (res, half, attended)

    def forward(self, pan, lms, ms):
        if self.variant == 'D0':
            return self.core(pan, lms, ms)
        if pan.shape != (len(lms), 1, *lms.shape[-2:]) or lms.shape[1] != 4 or ms.shape != (len(lms), 4, lms.shape[-2] // 4, lms.shape[-1] // 4):
            raise ValueError('QB NCHW inputs at scale 4 required')
        if lms.shape[-2] % 4 or lms.shape[-1] % 4:
            raise ValueError('LMS dimensions must be divisible by 4')
        current = lms
        previous = None
        for _ in range(2 if self.variant == 'D2' else 3):
            error = torch.zeros_like(current)
            if self.use_observation:
                observed = F.interpolate(current, size=ms.shape[-2:], mode='bilinear', align_corners=True)
                error = F.interpolate(ms - observed, size=current.shape[-2:], mode='bilinear', align_corners=True)
                error = torch.tanh(self.observation_gate).view(1, 4, 1, 1) * error
            corrected = current + self.error_encoder(error)
            if self.use_memory or self.use_pan_difference:
                proposal, previous = self._restore(pan, corrected, ms, previous)
            else:
                proposal = self.core(pan, corrected, ms)
            current = current + self.alpha * (proposal - current)
        return current
