"""Controlled SSA ablations; the official backbone and interpolation stay intact."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .ssa_mrn import RestoredPansharpeningNet


class AlignedSSA(nn.Module):
    def __init__(self, source, sigmoid=False):
        super().__init__()
        self.fixed = source.fixed
        self.sigmoid = sigmoid
        # Reuse every original parameter and its name, including unused layers.
        for name, child in source.named_children():
            self.add_module(name, child)

    def features(self, pan, ms):
        return self.conv6t6(self.relu(self.conv1t6(pan))), self.conv7t6_3(ms)

    def attention(self, pan, ms):
        p, q = self.features(pan, ms)
        logits = p * q
        weights = torch.sigmoid(logits) if self.sigmoid else logits.flatten(2).softmax(-1).reshape_as(p)
        return p, weights

    def forward(self, pan, ms):
        p, weights = self.attention(pan, ms)
        return p * weights


class LocalCrossSSA(AlignedSSA):
    """One head, K-dimensional dot product, masked 5x5 neighbors, PAN values."""
    def __init__(self, source, chunk_rows=32):
        super().__init__(source)
        self.value = nn.Conv2d(self.fixed, self.fixed, 1)
        self.chunk_rows = chunk_rows

    def attend(self, pan, ms, return_weights=False):
        k, q = self.features(pan, ms)
        v = self.value(k)
        b, c, h, w = q.shape
        outputs, maps = [], []
        # Accumulate logits and weighted values in FP32 under AMP.
        with torch.autocast(device_type=q.device.type, enabled=False):
            kp, vp = F.pad(k.float(), (2, 2, 2, 2)), F.pad(v.float(), (2, 2, 2, 2))
            mask = F.pad(q.new_ones(1, 1, h, w).float(), (2, 2, 2, 2))
            for top in range(0, h, self.chunk_rows):
                rows = min(self.chunk_rows, h-top)
                keys = F.unfold(kp[:, :, top:top+rows+4], 5).reshape(b, c, 25, rows*w)
                values = F.unfold(vp[:, :, top:top+rows+4], 5).reshape(b, c, 25, rows*w)
                valid = F.unfold(mask[:, :, top:top+rows+4], 5).reshape(1, 25, rows*w).bool()
                query = q[:, :, top:top+rows].float().reshape(b, c, 1, rows*w)
                logits = (query*keys).sum(1) / math.sqrt(c)
                weights = logits.masked_fill(~valid, -torch.inf).softmax(1)
                outputs.append((values*weights[:, None]).sum(2).reshape(b, c, rows, w))
                if return_weights:
                    maps.append(weights.reshape(b, 25, rows, w))
        output = torch.cat(outputs, 2).to(v.dtype)
        return (output, torch.cat(maps, 2)) if return_weights else output

    def forward(self, pan, ms):
        return self.attend(pan, ms)


class FusionPansharpeningNet(RestoredPansharpeningNet):
    def __init__(self, channels=4, ssai_dimension=6, variant="A0", chunk_rows=32):
        if variant not in ("A0", "A1", "A2", "A3"):
            raise ValueError("Unknown SSA variant")
        super().__init__(channels, ssai_dimension)
        self.fusion_variant = variant
        if variant == "A0":
            return
        # Preserve shared initialization even though A3 adds new value projections.
        with torch.random.fork_rng(devices=[]):
            for name in ("SSA_blocks", "SSA_blocks1", "SSA_blocks2"):
                blocks = getattr(self, name)
                for i, block in enumerate(blocks):
                    blocks[i] = LocalCrossSSA(block, chunk_rows) if variant == "A3" else AlignedSSA(block, variant == "A2")
