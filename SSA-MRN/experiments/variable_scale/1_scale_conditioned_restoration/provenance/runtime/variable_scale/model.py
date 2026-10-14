"""Same official SSA blocks; independent input ratio and internal pyramid."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .reference import ReferenceX4

SCALES = (2, 3, 4)

@lru_cache(maxsize=64)
def _axis_weights(length, target, align_corners, antialias, dtype, device):
    # CPU native interpolation of basis vectors gives the exact linear operator,
    # including the native edge normalization. Never approximate AA with another blur.
    basis = torch.eye(length, dtype=torch.float64).reshape(1, length, 1, length)
    weights = F.interpolate(basis, size=(1, target), mode="bilinear",
                            align_corners=align_corners, antialias=antialias)
    return weights[0, :, 0, :].t().contiguous().to(device=device, dtype=dtype)

class _DeterministicResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, height, width, align_corners, antialias):
        ctx.input_size = tuple(x.shape[-2:])
        ctx.align_corners, ctx.antialias = align_corners, antialias
        # Native forward is retained; only its CUDA scatter-based backward is replaced.
        return F.interpolate(x, size=(height, width), mode="bilinear",
                             align_corners=align_corners, antialias=antialias)

    @staticmethod
    def backward(ctx, gradient):
        h, w = ctx.input_size
        oh, ow = gradient.shape[-2:]
        wy = _axis_weights(h, oh, ctx.align_corners, ctx.antialias, gradient.dtype, gradient.device)
        wx = _axis_weights(w, ow, ctx.align_corners, ctx.antialias, gradient.dtype, gradient.device)
        # Dense transpose products avoid CUDA scatter/atomic accumulation. CUBLAS
        # workspace configuration is set before worker CUDA initialization by runner.
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            result = torch.matmul(torch.matmul(wy.t(), gradient), wx)
        return result, None, None, None, None

def deterministic_bilinear(x, size, align_corners=False, antialias=False):
    return _DeterministicResize.apply(x, *tuple(size), align_corners, antialias)

class DeterministicUpsample(nn.Upsample):
    """Reference-path resize with identical native forward and strict backward."""
    def forward(self, x):
        if x.requires_grad and torch.are_deterministic_algorithms_enabled():
            if self.size is not None:
                size = self.size
            else:
                factor = self.scale_factor
                factors = (factor, factor) if isinstance(factor, (int,float)) else factor
                size = tuple(int(s*f) for s,f in zip(x.shape[-2:], factors))
            return deterministic_bilinear(x, size, align_corners=self.align_corners)
        return super().forward(x)

def resize(x, size):
    size = tuple(size)
    if tuple(x.shape[-2:]) == size:
        return x
    antialias = any(a > b for a, b in zip(x.shape[-2:], size))
    if x.requires_grad and torch.are_deterministic_algorithms_enabled():
        return deterministic_bilinear(x, size, antialias=antialias)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=antialias)

class ScaleFiLM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, scale):
        condition = x.new_full((x.shape[0], 1), math.log2(scale))
        gamma, beta = self.net(condition).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class VariableScaleNet(ReferenceX4):
    def __init__(self, channels=4, conditioned=False):
        super().__init__(channels=channels, ssai_dimension=4)
        self.conditioned = conditioned
        # Created AFTER the reference layers: common initialization is identical.
        self.film = ScaleFiLM(64) if conditioned else nn.Identity()

    def forward(self, pan, lms, ms, scale):
        if not isinstance(scale, int) or isinstance(scale, bool) or scale not in SCALES:
            raise ValueError("Supported integer scales are 2, 3, 4")
        if any(x.ndim != 4 for x in (pan, lms, ms)):
            raise ValueError("Expected NCHW tensors")
        n, c, h, w = lms.shape
        if c != self.channels or min(h, w) < 12 or h % scale or w % scale:
            raise ValueError("Target dimensions must be divisible by input scale and >=12")
        if pan.shape != (n, 1, h, w) or ms.shape != (n, c, h // scale, w // scale):
            raise ValueError("PAN/LMS/MS shapes disagree with scale")
        half, quarter = (h // 2, w // 2), (h // 4, w // 4)
        # Original PAN low-pass context is at internal quarter scale, not input scale.
        context = self.prelu(self.conv1t1(resize(resize(pan, quarter), (h, w))))
        features = self.backbone(self.cov2t64(torch.cat((lms, pan, context), dim=1)))
        if self.conditioned:
            features = self.film(features, scale)
        res = self.conv64t48(features) + lms
        res = res + self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res_half, ms_half, pan_half = resize(res, half), resize(ms, half), resize(pan, half)
        res_half = res_half + self._attend(pan_half, res_half, ms_half,
                                         self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        ms_small = resize(ms, quarter)
        attended = self._attend(resize(pan, quarter), resize(res_half, quarter), ms_small,
                                self.SSA_blocks2, self.conv48tnum3, self.prelu3)
        out = self.conv2ctc1(torch.cat((ms_small, attended), dim=1))
        out = self.conv2ctc2(torch.cat((resize(out, half), res_half), dim=1))
        return self.conv2ctc3(torch.cat((resize(out, (h, w)), res), dim=1))
