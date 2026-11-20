"""90 arms: the 89 confirmed recipe (W2 2x2, lambda 1.2, c 0.25, 88 K2D: full-size 120 epochs -> K2 factorization -> 30-epoch cosine) with
the model-internal resampling of the official network (nn.Upsample bilinear, align_corners=True) replaced by fixed, non-learned operators.

Variants (no trainable parameter is added or removed; the kernels are non-persistent buffers, so state_dict and the seeded initial state
are those of the 89 2x2 model):
  N : unchanged (bilinear). The 89 confirmation model and the control of A, B and C.
  A : input upsampling with the dataset 23-tap interpolation (common/interpolation.interp23tap): upsample1 (x4, LRPAN) and upsample100
      (x2, MS to 1/2 size; the first stage of the x4 chain, so MS pixel j lands on full-size position 4j + 2 like the data).
  B : A + observation-matched LRPAN: downsample1 = PAN MTF blur (41x41, genMTF, the training sensor's PAN GNyq, replicated boundary)
      + x4 decimation at phase 2 (= 54 Wald.pan, the degradation that made the MS); then 23-tap x4 up (A).
  C : B + anti-aliased internal downsampling: downsample100 / downsample200 (x2, features and PAN) and downsample2 (x4, PAN) = the 23-tap
      half-band low-pass (DC gain 1, separable, periodic boundary) + decimation on the grid of the data (full -> 1/2 phase 0,
      1/2 -> 1/4 phase 1, full -> 1/4 = both, overall phase 2). The kernel is fixed before training and does not depend on the sensor.
The output upsampling (upsample101, upsample102) and the W2 branch 2x2 layers are not changed.

Arm names: <variant>T = full-size 2x2 C2MS trunk (120 epochs, lr 1e-4 constant), <variant>K = 83 K2 structure (2x2) from the <variant>T
epoch-120 weights (30-epoch cosine). Other names go to the 89 model (controls: 'L1S' = 65 C2MS-SP weights, 'K2D' = 88 K2D)."""
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp89_model', HERE.parent / '89_spectral_coarse_loss_weights/model.py')
M89 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M89)
M88, M83, M54, M43 = M89.M88, M89.M83, M89.M54, M89.M43
conv_macs, used_parameters, CAL_PATCHES, CAL_SEED, second_moments, factorize = M89.conv_macs, M89.used_parameters, M89.CAL_PATCHES, M89.CAL_SEED, M89.second_moments, M89.factorize
to_2x2, OUT_IMPL = M89.to_2x2, M89.OUT_IMPL

import sys
sys.path.insert(0, str(HERE.parents[2]))
from common.interpolation import Interp23  # noqa: E402

_gm_spec = importlib.util.spec_from_file_location('exp54_prepare_90', HERE.parent / '54_multi_sensor_scale_mixed_training/prepare.py')
P54 = importlib.util.module_from_spec(_gm_spec); _gm_spec.loader.exec_module(P54)

VARIANTS = ('N', 'A', 'B', 'C')
FULL_SIZE = tuple(v + 'T' for v in VARIANTS)
FACTORIZED = tuple(v + 'K' for v in VARIANTS)
SOURCE = {v + 'K': v + 'T' for v in VARIANTS}
ARMS = FULL_SIZE + FACTORIZED
PAN_GNYQ = {4: 0.15, 8: 0.14}  # 54 prepare SENSORS: QB 0.15, GF2 0.15 (toolbox default), WV3 0.14; the model only knows the band count
HALF = [.5, .305334091185, 0, -.072698593239, 0, .021809577942, 0, -.005192756653, 0, .000807762146, 0, -.000060081482]  # interp23tap


def halfband():
    """23-tap half-band low-pass with DC gain 1 (the interp23tap kernel / 2)."""
    return torch.tensor(HALF[:0:-1] + HALF, dtype=torch.float64)


class HalfDown(nn.Module):
    """x2 decimation after the separable 23-tap half-band low-pass, periodic boundary, keeping rows/cols phase::2."""

    def __init__(self, phase):
        super().__init__(); self.phase = phase; self.register_buffer('k', halfband().float(), persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            dt = x.dtype; x = x.float(); c = x.shape[1]; k = self.k.to(x.device)
            x = F.conv2d(F.pad(x, (0, 0, 11, 11), mode='circular'), k.view(1, 1, 23, 1).expand(c, 1, 23, 1), groups=c)
            x = F.conv2d(F.pad(x, (11, 11, 0, 0), mode='circular'), k.view(1, 1, 1, 23).expand(c, 1, 1, 23), groups=c)
            return x[..., self.phase::2, self.phase::2].to(dt)


class QuarterDown(nn.Module):
    """x4 = HalfDown(0) then HalfDown(1): full-size position 4j + 2 (phase 2, the grid of the data)."""

    def __init__(self):
        super().__init__(); self.a = HalfDown(0); self.b = HalfDown(1)

    def forward(self, x): return self.b(self.a(x))


class PanMTFDown(nn.Module):
    """PAN MTF blur (genMTF 41x41 at ratio 4, replicated boundary as 54 prepare._blur / Wald.pan) + x4 decimation at phase 2."""

    def __init__(self, gnyq):
        super().__init__(); self.gnyq = gnyq
        self.register_buffer('k', torch.from_numpy(P54.gen_mtf(gnyq, 4)).float()[None, None], persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            dt = x.dtype; x = x.float(); c = x.shape[1]; p = self.k.shape[-1] // 2; k = self.k.to(x.device).expand(c, 1, -1, -1)
            h, w = x.shape[-2:]
            rows = torch.arange(-p, h + p, device=x.device).clamp(0, h - 1); cols = torch.arange(-p, w + p, device=x.device).clamp(0, w - 1)
            x = F.conv2d(x.index_select(-2, rows).index_select(-1, cols), k, groups=c)
            return x[..., 2::4, 2::4].to(dt)


def apply_variant(m, variant, bands):
    """Replace the internal resampling modules of m.core in place (no RNG consumed, no parameter added)."""
    c = m.core
    if variant == 'N': return m
    if variant not in VARIANTS: raise ValueError(variant)
    c.upsample1 = Interp23(4); c.upsample100 = Interp23(2)
    if variant in ('B', 'C'): c.downsample1 = PanMTFDown(PAN_GNYQ[bands])
    if variant == 'C': c.downsample100 = HalfDown(0); c.downsample200 = HalfDown(1); c.downsample2 = QuarterDown()
    return m


def variant_of(arm): return arm[0] if arm in ARMS else None


def Model(arm, bands=4, seed=42):
    if arm in FULL_SIZE: return apply_variant(to_2x2(M54.Model('C2MS', bands, seed)), arm[0], bands)
    if arm in FACTORIZED: return apply_variant(to_2x2(M83.Model('K2', bands, seed)), arm[0], bands)
    return M89.Model(arm, bands, seed)
