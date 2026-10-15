"""SSA-MRN reconstructed from paper Fig.2-3 and equations (1)-(14).

Ambiguities are explicit in config.json; this is not the authors' verified runtime.
"""
import torch
from torch import nn
from torch.nn import functional as F


def resize(x, size):
    return F.interpolate(x, size=size, mode='bicubic', align_corners=False, antialias=False)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class BandAttention(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.spectral = nn.Conv2d(channels + 1, k, 3, padding=1)
        self.pan1 = nn.Conv2d(1, k, 3, padding=1)
        self.pan2 = nn.Conv2d(k, k, 3, padding=1)

    def forward(self, pan, features, band):
        spatial = self.pan2(F.relu(self.pan1(pan)))
        spectral = self.spectral(torch.cat((features, band), 1))
        p, q = spatial.flatten(2), spectral.flatten(2)
        # Eq.10: [N,K,HW] @ [N,HW,K] -> [N,K,K].
        weights = torch.softmax(p @ q.transpose(-1, -2), dim=-1)
        # Eq.12: matrix multiplication is required by the stated dimensions.
        return (weights @ p).reshape_as(spatial)


class AttentionStage(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.bands = nn.ModuleList(BandAttention(channels, k) for _ in range(channels))
        self.fusion = nn.Conv2d(channels * k, channels, 3, padding=1)
        self.activation = nn.PReLU()

    def forward(self, pan, features, ms):
        detail = torch.cat([block(pan, features, ms[:, i:i+1])
                            for i, block in enumerate(self.bands)], 1)
        return features + self.activation(self.fusion(detail))  # Eq.14 at EVERY scale.


class PaperSSAMRN(nn.Module):
    def __init__(self, channels=4, k=6):
        super().__init__()
        self.channels, self.k = channels, k
        self.lrpan = nn.Conv2d(1, 1, 3, padding=1)
        self.stem = nn.Conv2d(channels + 2, 64, 3, padding=1)
        self.backbone = nn.Sequential(*(ResidualBlock() for _ in range(4)))
        self.project = nn.Conv2d(64, channels, 3, padding=1)
        self.stages = nn.ModuleList(AttentionStage(channels, k) for _ in range(3))
        self.decoder = nn.ModuleList(nn.Conv2d(2 * channels, channels, 3, padding=1) for _ in range(3))

    def forward(self, pan, supplied_lms, ms):
        n, c, h, w = supplied_lms.shape
        if c != self.channels or h % 4 or w % 4 or pan.shape != (n, 1, h, w) or ms.shape != (n, c, h//4, w//4):
            raise ValueError('Expected PAN/LMS at H,W and MS at H/4,W/4')
        # Paper requires bicubic LMS; supplied H5 LMS is deliberately not used.
        lms = resize(ms, (h, w))
        pl = resize(resize(pan, (h//4, w//4)), (h, w))
        f = self.project(self.backbone(self.stem(torch.cat((lms, pan, F.relu(self.lrpan(pl))), 1))))
        # Fig.2 has no additional LMS residual around the feature extractor.
        full = self.stages[0](pan, f, lms)
        half = self.stages[1](resize(pan, (h//2, w//2)), resize(full, (h//2, w//2)), resize(ms, (h//2, w//2)))
        small = self.stages[2](resize(pan, (h//4, w//4)), resize(half, (h//4, w//4)), ms)
        out = self.decoder[0](torch.cat((ms, small), 1))
        out = self.decoder[1](torch.cat((resize(out, (h//2, w//2)), half), 1))
        return self.decoder[2](torch.cat((resize(out, (h, w)), full), 1))
