"""Isolated RGB07 ablations; legacy checkpoints keep their original operations."""
import torch
from torch import nn
from torch.nn import functional as F
from .rgb_grouped import RGBTripleJointSsaMRN, ScalarGroupedCore
from .interp23 import interp23tap


class PilotCore(ScalarGroupedCore):
    def __init__(self, attention_mode='upstream'):
        super().__init__(17)
        if attention_mode not in ('upstream', 'aligned_spatial'):
            raise ValueError('Unknown pilot attention mode')
        self.attention_mode = attention_mode

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        if self.attention_mode == 'upstream':
            return super()._attend(pan, features, ms, blocks, fusion, activation)
        n, _, h, w = pan.shape
        guide = F.conv2d(pan, torch.cat([b.conv1t6.weight for b in blocks]),
                         torch.cat([b.conv1t6.bias for b in blocks]), padding=1).relu()
        guide = F.conv2d(guide, torch.cat([b.conv6t6.weight for b in blocks]),
                         torch.cat([b.conv6t6.bias for b in blocks]), padding=1, groups=17)
        projected = F.conv2d(features, torch.cat([b.conv7t6_3.weight[:, :17] for b in blocks]), padding=1)
        projected = projected + F.conv2d(ms, torch.cat([b.conv7t6_3.weight[:, 17:] for b in blocks]),
                         torch.cat([b.conv7t6_3.bias for b in blocks]), padding=1, groups=17)
        # Only remove the coordinate transpositions; keep spatial softmax unchanged.
        attention = (guide * projected).flatten(2).softmax(-1)
        return activation(fusion((guide.flatten(2) * attention).reshape(n, 68, h, w)))


class GroupGlobalEncoder(nn.Module):
    def __init__(self, grouped):
        super().__init__()
        self.grouped = grouped
        self.global_features = nn.Conv2d(204, 17, 1)
        self.mix = nn.Conv2d(34, 17, 1)
        # Begin with the baseline grouped representation; learn cross-group mixing.
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        with torch.no_grad():
            for i in range(17):
                self.mix.weight[i, i, 0, 0] = 1

    def forward(self, lr):
        return self.mix(torch.cat([self.grouped(lr), self.global_features(lr)], 1))


def rgb_highpass(rgb):
    low = interp23tap(F.avg_pool2d(rgb, 4), 4)
    return rgb - low, low


class GatedRGBDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 17, 3, padding=1), nn.ReLU())
        self.gate = nn.Conv2d(34, 17, 1)
        self.decode = nn.Conv2d(17, 204, 1)
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)

    def forward(self, rgb, hsi_features):
        detail = self.features(rgb_highpass(rgb)[0])
        confidence = self.gate(torch.cat([detail, hsi_features], 1)).sigmoid()
        return self.decode(detail * confidence)


class LocalRGBAlignment(nn.Module):
    """LR feature-conditioned offsets in HR pixels, plus spatial confidence."""
    def __init__(self, max_shift=2.):
        super().__init__()
        self.max_shift = float(max_shift)
        self.rgb_features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.hsi_features = nn.Sequential(nn.Conv2d(17, 16, 3, padding=1), nn.ReLU())
        self.predict = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 3, padding=1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, rgb, latent):
        fields = self.predict(torch.cat([self.rgb_features(F.avg_pool2d(rgb, 4)),
                                         self.hsi_features(latent)], 1))
        fields = F.interpolate(fields, size=rgb.shape[-2:], mode='bilinear', align_corners=False).float()
        offset = fields[:, :2].tanh() * self.max_shift
        # 2*sigmoid gives confidence=1 at initialization, preserving the RGB07 guide.
        confidence = 2 * fields[:, 2:3].sigmoid()
        n, _, h, w = rgb.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=rgb.device, dtype=torch.float32),
                                torch.arange(w, device=rgb.device, dtype=torch.float32), indexing='ij')
        grid = torch.stack([2 * (xx + .5) / w - 1, 2 * (yy + .5) / h - 1], -1)[None].expand(n, -1, -1, -1)
        delta = offset.permute(0, 2, 3, 1) * offset.new_tensor([2 / w, 2 / h])
        # Cross-modal warping in float32 avoids half-precision coordinate rounding.
        with torch.autocast(rgb.device.type, enabled=False):
            warped = F.grid_sample(rgb.float(), grid + delta, padding_mode='border', align_corners=False)
        high, low = rgb_highpass(warped)
        return low + confidence * high


class RGBPilotJoint17(RGBTripleJointSsaMRN):
    def __init__(self, options=None):
        super().__init__()
        options = options or {}
        unknown = set(options) - {'attention_mode', 'detail_path', 'global_encoder', 'local_alignment', 'max_shift'}
        if unknown:
            raise ValueError('Unknown pilot options: ' + str(sorted(unknown)))
        attention = options.get('attention_mode', 'upstream')
        # Replace each core using the identical parameter values, preserving initialization.
        for i, core in enumerate(self.cores):
            replacement = PilotCore(attention)
            replacement.load_state_dict(core.state_dict())
            self.cores[i] = replacement
        if options.get('global_encoder'):
            self.encoder = GroupGlobalEncoder(self.encoder)
        self.detail_path = GatedRGBDetail() if options.get('detail_path') else None
        self.local_alignment = LocalRGBAlignment(options.get('max_shift', 2.)) if options.get('local_alignment') else None

    def forward(self, rgb, lr_hsi):
        if rgb.ndim != 4 or lr_hsi.ndim != 4 or rgb.shape[1] != 3 or lr_hsi.shape[1] != 204:
            raise ValueError('Expected NCHW RGB=3 and HSI=204')
        if rgb.shape[0] != lr_hsi.shape[0] or rgb.shape[-2:] != tuple(s * 4 for s in lr_hsi.shape[-2:]):
            raise ValueError('Expected matching batches at x4 spatial ratio')
        latent = self.encoder(lr_hsi)
        up = interp23tap(latent, 4)
        guide = self.local_alignment(rgb, latent) if self.local_alignment is not None else rgb
        features = torch.cat([core(guide[:, i:i+1], up, latent) for i, core in enumerate(self.cores)], 1)
        correction = self.decoder(features)
        if self.detail_path is not None:
            correction = correction + self.detail_path(guide, up)
        return interp23tap(lr_hsi, 4) + correction
