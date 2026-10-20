"""Differentiable RR MS observation using the MATLAB DLPan MTF protocol.

DLPan-Toolbox at a34f884af88967580e3ef4f61a89e12fdfe4cda5:
Tools/genMTF.m, MTF.m, resize_images.m. Radial Kaiser window follows
MathWorks fwind1's documented Huang construction. Training-only audits
verify sampling and patch interiors; this does not establish FR sensor physics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = "a34f884af88967580e3ef4f61a89e12fdfe4cda5"
GNYQ = {
    "QB": [0.34, 0.32, 0.30, 0.22],
    "WV3": [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315],
}


def module_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def mtf_kernels(sensor, ratio=4, size=41):
    """Return MATLAB-protocol real FIR kernels (C,1,N,N), including signed taps."""
    if sensor not in GNYQ:
        raise ValueError(f"No sourced GNyq coefficients for {sensor}; do not assume GF2=IKONOS")
    if ratio != 4 or size != 41:
        raise ValueError("Only audited x4, 41-tap kernels supported")
    coordinate = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    radial_window = np.interp(np.sqrt(x*x+y*y), np.arange(size//2+1),
                              np.kaiser(size, 0.5)[size//2:], right=0.0)
    kernels = []
    for gain in GNYQ[sensor]:
        alpha = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(gain)))
        desired = np.exp(-(x*x+y*y)/(2*alpha*alpha))
        desired[desired < np.finfo(desired.dtype).eps*desired.max()] = 0
        desired /= desired.max()
        # MATLAB fwind1: inverse frequency response times radial 1-D window.
        # Keep signed taps and DC gain; clipping/renormalizing changes the operator.
        impulse = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * radial_window
        kernels.append(np.real(impulse))
    return np.stack(kernels)[:, None].astype(np.float64)


class SensorObservation(nn.Module):
    def __init__(self, sensor, sampling="decimate", phase=(2, 2), ratio=4):
        super().__init__()
        if sampling not in ("decimate", "bicubic"):
            raise ValueError("Unknown sampling")
        if len(phase) != 2 or any(v not in range(ratio) for v in phase):
            raise ValueError("Invalid sampling phase")
        self.sensor, self.sampling, self.phase, self.ratio = sensor, sampling, tuple(phase), ratio
        self.register_buffer("kernel", torch.from_numpy(mtf_kernels(sensor, ratio)))

    def blur(self, prediction):
        if prediction.ndim != 4 or prediction.shape[1] != self.kernel.shape[0]:
            raise ValueError("Expected NCHW with the sensor's bands")
        if any(n % self.ratio for n in prediction.shape[-2:]):
            raise ValueError("Spatial dimensions must be divisible by ratio")
        # Replication implemented using index_select: deterministic CUDA backward,
        # unlike replication_pad2d backward on some PyTorch versions.
        padding = self.kernel.shape[-1] // 2
        rows = torch.arange(-padding, prediction.shape[-2]+padding, device=prediction.device).clamp(0,prediction.shape[-2]-1)
        cols = torch.arange(-padding, prediction.shape[-1]+padding, device=prediction.device).clamp(0,prediction.shape[-1]-1)
        padded = prediction.index_select(-2, rows).index_select(-1, cols)
        return F.conv2d(padded, self.kernel.to(dtype=prediction.dtype), groups=prediction.shape[1])

    def sample(self, blurred):
        if self.sampling == "bicubic":
            return F.interpolate(blurred, scale_factor=1/self.ratio, mode="bicubic", align_corners=False, antialias=False)
        return blurred[..., self.phase[0]::self.ratio, self.phase[1]::self.ratio]

    def forward(self, prediction):
        return self.sample(self.blur(prediction))

    def loss(self, prediction, ms, margin, phases=None):
        if phases is None:
            observed = self(prediction)
        else:
            blurred = self.blur(prediction)
            if len(phases) != len(prediction):
                raise ValueError("Phase batch mismatch")
            # Phases are fixed by original TRAIN pairs, never inferred from predictions.
            observed = torch.cat([blurred[i:i+1, :, int(p[0])::self.ratio, int(p[1])::self.ratio]
                                  for i, p in enumerate(phases)], dim=0)
        if observed.shape != ms.shape:
            raise ValueError("Observed prediction and MS shape differ")
        if margin < 0 or min(ms.shape[-2:]) <= 2*margin:
            raise ValueError("Invalid audited LR interior margin")
        if margin:
            observed = observed[..., margin:-margin, margin:-margin]
            ms = ms[..., margin:-margin, margin:-margin]
        return F.mse_loss(observed, ms)


def load_profile(path, expected_sha256=None, sensor=None, train_path=None):
    path = Path(path)
    content = path.read_bytes()
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Observation profile hash changed")
    profile = json.loads(content)
    if profile.get("status") != "validated":
        raise ValueError("Observation data audit has not passed")
    if profile.get("operator_sha256") != module_hash():
        raise ValueError("Observation code changed since audit")
    if sensor and profile["sensor"] != sensor:
        raise ValueError("Observation profile sensor mismatch")
    if train_path and Path(profile["train_path"]).resolve() != Path(train_path).resolve():
        raise ValueError("Observation profile dataset mismatch")
    stat = Path(profile["train_path"]).stat()
    if stat.st_size != profile["train_stat"]["size"] or stat.st_mtime_ns != profile["train_stat"]["mtime_ns"]:
        raise ValueError("Training file changed since observation audit")
    if profile.get("phase_policy") != "fixed_per_training_sample" or len(profile.get("phases",[])) != profile["sample_count"] or profile.get("coverage") != profile["sample_count"]:
        raise ValueError("Training phase manifest incomplete")
    if any(list(p) not in [[2,2],[1,2],[2,1]] for p in profile["phases"]):
        raise ValueError("Unverified phase in manifest")
    operator = SensorObservation(profile["sensor"],profile["sampling"],ratio=profile["ratio"])
    return operator, profile
