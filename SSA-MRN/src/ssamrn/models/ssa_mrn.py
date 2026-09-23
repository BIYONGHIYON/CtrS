"""Minimal forward-path repair around the untouched official network.py.

The official forward(pan, lms) references an undefined `ms`. Its three
spatial stages require PAN and interpolated LMS at full resolution, plus
the original MS at one-quarter resolution. This adapter supplies that input
explicitly and retains the original modules, parameter names and operations.
"""

import importlib.util
from pathlib import Path

import torch


UPSTREAM_PATH = Path(__file__).resolve().parents[3] / "references/upstream/network.py"


def _upstream_model():
    if not UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"Initialize the official SSA-MRN submodule: {UPSTREAM_PATH}")
    spec = importlib.util.spec_from_file_location("ssamrn_official_network", UPSTREAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PansharpeningNet


class RestoredPansharpeningNet(_upstream_model()):
    """Official layers with an explicit low-resolution MS input."""

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
