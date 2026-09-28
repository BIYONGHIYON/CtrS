"""Minimal forward-path repair around the untouched official network.py.

The official forward(pan, lms) references an undefined `ms`. Its three
spatial stages require PAN and interpolated LMS at full resolution, plus
the original MS at one-quarter resolution. This adapter supplies that input
explicitly and retains the original modules, parameter names and operations.
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn


UPSTREAM_PATH = Path(__file__).resolve().parents[3] / "references/upstream/network.py"


def _upstream_model():
    if not UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"Initialize the official SSA-MRN submodule: {UPSTREAM_PATH}")
    spec = importlib.util.spec_from_file_location("ssamrn_official_network", UPSTREAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PansharpeningNet


class DirectMLPReLU(nn.PReLU):
    """Equivalent PReLU expression avoiding a DirectML batch backward error."""

    def forward(self, input):
        slope = self.weight.view(1, -1, 1, 1)
        return torch.where(input > 0, input, slope * input)


class RestoredPansharpeningNet(_upstream_model()):
    """Official layers with an explicit low-resolution MS input."""

    def __init__(self, channels, ssai_dimension=4):
        super().__init__(channels)
        if ssai_dimension < 1:
            raise ValueError("ssai_dimension must be positive")
        self.ssai_dimension = ssai_dimension
        if ssai_dimension == 4:
            return  # Preserve the published upstream architecture and existing checkpoints.

        # The upstream code hard-codes K=4, while the paper specifies K=6.
        # Resize only the layers used by the restored forward path; leave upstream untouched.
        self.fixed = ssai_dimension
        for blocks in (self.SSA_blocks, self.SSA_blocks1, self.SSA_blocks2):
            for block in blocks:
                block.fixed = ssai_dimension
                block.conv1t6 = nn.Conv2d(1, ssai_dimension, 3, padding=1)
                block.conv6t6 = nn.Conv2d(ssai_dimension, ssai_dimension, 3, padding=1)
                block.conv7t6_3 = nn.Conv2d(channels + 1, ssai_dimension, 3, padding=1)
        for name in ("conv48tnum1", "conv48tnum2", "conv48tnum3"):
            setattr(self, name, nn.Conv2d(ssai_dimension * channels, channels, 3, padding=1))

    def use_directml_prelu(self):
        """Keep PReLU weights/checkpoints while using a DirectML-safe backward."""
        for name, module in self.named_modules():
            if not name or not isinstance(module, nn.PReLU):
                continue
            parent = self.get_submodule(name.rpartition(".")[0]) if "." in name else self
            replacement = DirectMLPReLU(module.num_parameters)
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
            setattr(parent, name.rpartition(".")[2], replacement)

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        outputs = []
        for band in range(self.channels):
            band_input = torch.cat([features, ms[:, band : band + 1]], dim=1)
            outputs.append(blocks[band](pan, band_input))
        return activation(fusion(torch.cat(outputs, dim=1)))

    def forward(self, pan, lms, ms):
        if pan.ndim != 4 or lms.ndim != 4 or ms.ndim != 4:
            raise ValueError("Expected NCHW tensors for pan, lms, and ms")
        n, channels, h, w = lms.shape
        if channels != self.channels or h % 4 or w % 4 or min(h, w) < 4:
            raise ValueError(f"Invalid LMS shape: {tuple(lms.shape)}")
        if pan.shape != (n, 1, h, w) or ms.shape != (n, channels, h // 4, w // 4):
            raise ValueError("Expected PAN at LMS resolution and MS at 1/4 resolution")

        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        features = torch.cat([lms, pan, pan_updown], dim=1)
        features = self.backbone(self.cov2t64(features))
        res = self.conv64t48(features) + lms

        pan_ms = self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res = res + pan_ms

        res_half = self.downsample100(res)
        ms_half = self.upsample100(ms)
        pan_half = self.downsample100(pan)
        pan_ms = self._attend(pan_half, res_half, ms_half, self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        res_half = res_half + pan_ms

        res_quarter = self.downsample200(res_half)
        pan_quarter = self.downsample2(pan)
        pan_ms = self._attend(pan_quarter, res_quarter, ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)

        output = self.conv2ctc1(torch.cat([ms, pan_ms], dim=1))
        output = self.conv2ctc2(torch.cat([self.upsample101(output), res_half], dim=1))
        output = self.conv2ctc3(torch.cat([self.upsample102(output), res], dim=1))
        return output
