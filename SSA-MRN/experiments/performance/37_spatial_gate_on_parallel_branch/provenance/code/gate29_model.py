"""Experiment 29 W2, with an optional spatial gate on its added feature branch.

Original W2 classes are shipped separately, frozen from the archived source.
The gate changes only the addition inside Parallel.forward, after PixelShuffle.
"""
import torch
from torch import nn
from common.model import RestoredPansharpeningNet
from original_capacity import Parallel


class GatedParallel(Parallel):
    def __init__(self, original):
        super().__init__(original, low=True)
        # Isolate the new parameters' RNG from the common model initialization.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Sequential(nn.Conv2d(128, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        self.last_gate = None

    def forward(self, x):
        b = self.original(x)
        y = self.branch(x)
        g = self.gate(torch.cat((b, y), dim=1))
        if not self.training:
            self.last_gate = g.detach()
            self.ratio = (y.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)).detach()
        return b + g*y


class W2Net(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self.gated = gated
        self.net = RestoredPansharpeningNet(4, 4)
        self.net.backbone = GatedParallel(self.net.backbone) if gated else Parallel(self.net.backbone, low=True)

    def forward(self, pan, lms, ms):
        return self.net(pan, lms, ms)


def common_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('net.backbone.gate.')}


def load_common(model, state):
    result = model.load_state_dict(state, strict=False)
    expected = {k for k in model.state_dict() if k.startswith('net.backbone.gate.')}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError('Common initialization keys differ: '+str(result))
