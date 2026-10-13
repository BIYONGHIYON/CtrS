import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""GT-only multiscale readouts; the restored SSA-MRN forward stays intact."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from common.model import RestoredPansharpeningNet

DEFAULT_AUX_WEIGHTS = {"quarter": 0.005, "half": 0.005, "full": 0.001}


def target_spec():
    return {"source": "normalized GT only; never observed LRMS or Haar LL",
            "ratios": {"quarter": 4, "half": 2, "full": 1},
            "filter": "separable positive unit-sum binomial FIR",
            "binomial_orders": {"quarter": 16, "half": 4},
            "coefficients": {name: [math.comb(order, i) / 2**order for i in range(order+1)]
                             for name, order in (("quarter", 16), ("half", 4))},
            "boundary": "replicate on both axes",
            "coordinates": "j*(N-1)/(M-1), M=N/ratio; bilinear align_corners=True",
            "full": "identity", "precision": "FP32",
            "interpretation": "explicit synthetic GT targets, not a calibrated sensor MTF"}


@lru_cache(maxsize=32)
def _target_geometry(height, width, channels, ratio, device):
    order = ratio**2
    kernel = torch.tensor([math.comb(order, i) / 2**order for i in range(order+1)],
                          device=device, dtype=torch.float32)
    radius = order // 2
    rows = torch.arange(-radius, height+radius, device=device).clamp(0, height-1)
    cols = torch.arange(-radius, width+radius, device=device).clamp(0, width-1)
    horizontal = kernel[None,None,None,:].expand(channels,1,1,-1).contiguous()
    vertical = kernel[None,None,:,None].expand(channels,1,-1,1).contiguous()
    return rows, cols, horizontal, vertical


def gt_target(gt, ratio):
    if gt.ndim != 4 or not gt.is_floating_point() or ratio not in (1, 2, 4):
        raise ValueError("Expected floating NCHW GT and ratio 1, 2 or 4")
    if any(n % ratio for n in gt.shape[-2:]):
        raise ValueError("GT dimensions must be divisible by the target ratio")
    with torch.autocast(gt.device.type, enabled=False):
        x = gt.float()
        if ratio == 1:
            return x
        channels = x.shape[1]
        # index_select reproduces replicate padding and has deterministic CUDA backward.
        rows, cols, horizontal, vertical = _target_geometry(*x.shape[-2:], channels, ratio, x.device)
        x = F.conv2d(x.index_select(-1, cols), horizontal, groups=channels)
        x = F.conv2d(x.index_select(-2, rows), vertical, groups=channels)
        return F.interpolate(x, size=(gt.shape[-2]//ratio, gt.shape[-1]//ratio),
                             mode="bilinear", align_corners=True)


class PyramidSupervision(nn.Module):
    def __init__(self, channels=4, k=4):
        super().__init__()
        self.core = RestoredPansharpeningNet(channels, k)
        self.aux_heads = nn.ModuleDict({"readout_"+name: nn.Conv2d(channels, channels, 1)
                                       for name in ("quarter", "half", "full")})
        with torch.no_grad():
            for head in self.aux_heads.values():
                head.weight.copy_(torch.eye(channels).reshape(channels, channels, 1, 1))
                head.bias.zero_()

    def forward(self, pan, lms, ms, return_aux=False):
        if not return_aux:
            return self.core(pan, lms, ms)
        captures = {}
        handles = []
        try:
            for name, layer in (("quarter", "conv2ctc1"), ("half", "conv2ctc2"), ("full", "conv2ctc3")):
                handles.append(getattr(self.core, layer).register_forward_hook(
                    lambda module, inputs, output, name=name: captures.__setitem__(name, output)))
            final = self.core(pan, lms, ms)
            auxiliary = {name: self.aux_heads["readout_"+name](output) for name, output in captures.items()}
            return final, auxiliary
        finally:
            for handle in handles:
                handle.remove()


def build_model(variant, channels, k):
    if variant not in ("B0", "B0_DS"):
        raise ValueError(variant)
    return PyramidSupervision(channels, k)


def training_loss(model, inputs, gt, weights):
    if set(weights) != {"quarter", "half", "full"} or any(
            not math.isfinite(float(v)) or v < 0 for v in weights.values()):
        raise ValueError("Three finite nonnegative auxiliary weights are required")
    enabled = any(weights.values())
    result = model(*inputs, return_aux=enabled)
    prediction, auxiliary = result if enabled else (result, {})
    mse = F.mse_loss(prediction.float(), gt.float())
    total = mse
    components = {"final_mse": mse}
    for name, ratio in (("quarter", 4), ("half", 2), ("full", 1)):
        term = F.l1_loss(auxiliary[name].float(), gt_target(gt, ratio)) if enabled else mse.new_zeros(())
        components["aux_"+name+"_l1"] = term
        total = total + weights[name] * term
    components["objective"] = total
    return prediction, total, components


def train_config(variant, seed):
    return {"variant": variant, "seed": seed, "core_passes": 1,
            "head_sites": ["conv2ctc1", "conv2ctc2", "conv2ctc3"],
            "loss": "final MSE + weighted mean auxiliary L1",
            "final_output": "original conv2ctc3 output; auxiliary heads training only",
            "target_spec": target_spec()}
