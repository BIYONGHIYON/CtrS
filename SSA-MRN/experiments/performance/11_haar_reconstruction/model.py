
import torch
from torch import nn
from torch.nn import functional as F

def dwt(x):
    # Orthonormal Haar, top-left origin. Directions defined by these signs.
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a+b+c+d)*0.5, torch.cat(((a-b+c-d)*0.5,
               (a+b-c-d)*0.5, (a-b-c+d)*0.5), dim=1)

def iwt(low, high):
    lh, hl, hh = high.chunk(3, dim=1)
    # Pixel shuffle expects four subpixels contiguous per output band.
    pixels = torch.stack((low+lh+hl+hh, low-lh+hl-hh,
                          low+lh-hl-hh, low-lh-hl+hh), dim=2)*0.5
    return F.pixel_shuffle(pixels.flatten(1, 2), 2)

class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))
    def forward(self, x): return x + self.body(x)*0.1

class Stage(nn.Module):
    def __init__(self, bands, width, blocks):
        super().__init__()
        self.ms = nn.Sequential(nn.Conv2d(bands, width, 3, padding=1), nn.SiLU(),
                                *[ResBlock(width) for _ in range(blocks)])
        self.pan = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(), ResBlock(width))
        self.joint = nn.Sequential(nn.Conv2d(2*width, width, 3, padding=1), nn.SiLU(),
                                   ResBlock(width))
        self.low = nn.Conv2d(width, bands, 3, padding=1)
        self.high_ms = nn.Conv2d(width, 3*bands, 3, padding=1)
        self.residual = nn.Conv2d(width, 3*bands, 3, padding=1)
        # PAN injection initially zero. All bands are mixed by regular convolutions.
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.low.weight); nn.init.zeros_(self.low.bias)
    def forward(self, x, pan_coeff, gate_head=None, diagnostics=False):
        fm, fp = self.ms(x), self.pan(pan_coeff)
        condition = torch.cat((fm, fp), 1)
        low = 2*x + self.low(fm)  # Learned LL; observed MS is not assumed GT LL.
        base = self.high_ms(fm)
        residual = self.residual(self.joint(condition))
        gate = None if gate_head is None else gate_head(condition).sigmoid()
        injected = residual if gate is None else gate*residual
        output = iwt(low, base + injected)
        if not diagnostics: return output
        if gate is None: gate = torch.ones_like(residual)
        # high channels are direction-major: [LH bands, HL bands, HH bands].
        shape = (len(x), 3, x.shape[1], *x.shape[-2:])
        g, r, j, m = [t.float().reshape(shape) for t in (gate, residual, injected, base)]
        axes = (0, 3, 4)
        stats = torch.stack((g.mean(axes), g.std(axes, unbiased=False),
             (g<0.05).float().mean(axes), (g>0.95).float().mean(axes),
             r.square().mean(axes), j.square().mean(axes), m.square().mean(axes)))
        return output, stats

class WaveletMS(nn.Module):
    def __init__(self, variant, bands=4, width=64, blocks=3):
        super().__init__()
        if variant not in ("A0", "A1"): raise ValueError(variant)
        self.variant = variant
        # Instantiate ALL shared modules before gates so same-seed weights match.
        self.stages = nn.ModuleList([Stage(bands, width, blocks) for _ in range(2)])
        self.gates = nn.ModuleList()
        if variant == "A1":
            for _ in range(2):
                head = nn.Conv2d(2*width, 3*bands, 1)
                nn.init.zeros_(head.weight); nn.init.constant_(head.bias, -2.)
                self.gates.append(head)
    def forward(self, pan, ms, diagnostics=False):
        fine_l, fine_h = dwt(pan)
        coarse_l, coarse_h = dwt(fine_l)
        coefficients = (torch.cat((coarse_l, coarse_h), 1), torch.cat((fine_l, fine_h), 1))
        x = ms; stats = []
        for s, stage in enumerate(self.stages):
            gate = self.gates[s] if self.variant == "A1" else None
            result = stage(x, coefficients[s], gate, diagnostics)
            if diagnostics: x, value = result; stats.append(value)
            else: x = result
        return (x, torch.stack(stats)) if diagnostics else x
