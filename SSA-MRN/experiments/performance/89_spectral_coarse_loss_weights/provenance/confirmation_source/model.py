"""90 arms: the 89 confirmed recipe (W2 2x2, lambda 1.2, c 0.25, 88 K2D: full-size 120 epochs -> K2 factorization -> 30-epoch cosine) with
the model-internal resampling of the official network (nn.Upsample bilinear, align_corners=True) replaced by fixed, non-learned operators
of the dataset (23-tap interpolation, sensor MTF blur + decimation).

Variants (no trainable parameter is added or removed; the kernels are non-persistent buffers, so state_dict and the seeded initial state
are those of the 89 2x2 model):
  N : unchanged (bilinear). The 89 confirmation model and the control of A, B and C.
  A : input upsampling with the dataset 23-tap interpolation: upsample1 (x4, LRPAN) and upsample100 (x2, MS to 1/2 size, the first stage
      of the x4 chain, so MS pixel j lands on full-size position 4j + 2 like the data).
  B : A + observation-matched LRPAN: downsample1 = PAN MTF blur (genMTF 41x41 at ratio 4, the training sensor's PAN GNyq, replicated
      boundary) + x4 decimation at phase 2 (= 54 Wald.pan, the degradation that made the MS); then 23-tap x4 up (A).
  C : every fixed resampling of the model matched to the dataset (user decision 2026-11-20 23h):
      B + internal downsampling with the sensor MTF: downsample100 (x2: features with the MS MTF of each band, PAN with the PAN MTF, both
      genMTF at ratio 2 = the 2x coarse-pair kernels) at phase 0, downsample200 (features x2, MS MTF ratio 2) at phase 1, downsample2
      (PAN x4, PAN MTF ratio 4) at phase 2; the phases follow the data grid (1/2-size pixel k = full-size 2k, 1/4-size pixel j = 4j + 2)
      + output upsampling with the 23-tap interpolation: upsample101 (1/4 -> 1/2, first stage, insertion phase 1) and upsample102
      (1/2 -> full, second stage, insertion phase 0), i.e. the two stages of the dataset x4 interpolation.
The W2 branch 2x2 layers are learned and not changed. The MTF of the training sensor is used (on the WV2 transfer the WV3 kernels stay).

Arm names: <variant>T = full-size 2x2 C2MS trunk (120 epochs, lr 1e-4 constant), <variant>K = 83 K2 structure (2x2) from the <variant>T
epoch-120 weights (30-epoch cosine). Other names go to the 89 model (controls: 'L1S' = 65 C2MS-SP weights, 'K2D' = 88 K2D).
B and C need the training sensor: Model(..., sensor=) or SENSOR['value'] (set by train.py / evaluate.py / run.py)."""
import importlib.util
import sys
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

sys.path.insert(0, str(HERE.parents[2]))
from common.interpolation import Interp23  # noqa: E402

_gm_spec = importlib.util.spec_from_file_location('exp54_prepare_90', HERE.parent / '54_multi_sensor_scale_mixed_training/prepare.py')
P54 = importlib.util.module_from_spec(_gm_spec); _gm_spec.loader.exec_module(P54)

VARIANTS = ('N', 'A', 'B', 'C')
FULL_SIZE = tuple(v + 'T' for v in VARIANTS)
FACTORIZED = tuple(v + 'K' for v in VARIANTS)
SOURCE = {v + 'K': v + 'T' for v in VARIANTS}
ARMS = FULL_SIZE + FACTORIZED
SENSOR = {'value': None}  # training sensor for the MTF kernels of B and C
HALF = [.5, .305334091185, 0, -.072698593239, 0, .021809577942, 0, -.005192756653, 0, .000807762146, 0, -.000060081482]  # interp23tap


class Up23(nn.Module):
    """One x2 stage of the dataset 23-tap interpolation (common/interpolation.interp23tap): zero insertion at `offset`::2, separable
    23-tap filter (gain 2), periodic boundary. Up23(0)(Up23(1)(x)) == interp23tap(x, 4); Up23(1) == interp23tap(x, 2)."""

    def __init__(self, offset):
        super().__init__(); self.offset = offset
        self.register_buffer('k', torch.tensor(HALF[:0:-1] + HALF, dtype=torch.float32) * 2, persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            dt = x.dtype; x = x.float(); n, c, h, w = x.shape; k = self.k.to(x.device)
            up = x.new_zeros(n, c, 2 * h, 2 * w); up[:, :, self.offset::2, self.offset::2] = x
            rows = torch.arange(-11, 2 * h + 11, device=x.device) % (2 * h); cols = torch.arange(-11, 2 * w + 11, device=x.device) % (2 * w)
            y = F.conv2d(up.index_select(2, rows), k.view(1, 1, 23, 1).expand(c, 1, 23, 1).contiguous(), groups=c)
            y = F.conv2d(y.index_select(3, cols), k.view(1, 1, 1, 23).expand(c, 1, 1, 23).contiguous(), groups=c)
            return y.to(dt)


def _blur_rep(x, k):
    """Depthwise blur with replicated boundary (54 prepare._blur)."""
    p = k.shape[-1] // 2; h, w = x.shape[-2:]
    rows = torch.arange(-p, h + p, device=x.device).clamp(0, h - 1); cols = torch.arange(-p, w + p, device=x.device).clamp(0, w - 1)
    return F.conv2d(x.index_select(-2, rows).index_select(-1, cols), k, groups=x.shape[1])


class MTFDown(nn.Module):
    """Sensor MTF blur (genMTF 41x41 at `ratio`) + decimation by `ratio` at `phase`. A 1-channel input is PAN (PAN GNyq), an input with the
    sensor's band count is MS-like (per-band MS GNyq)."""

    def __init__(self, sensor, ratio, phase):
        super().__init__(); s = P54.SENSORS[sensor]; self.sensor, self.ratio, self.phase = sensor, ratio, phase
        self.register_buffer('kp', torch.from_numpy(P54.gen_mtf(s['pan_gnyq'], ratio)).float()[None, None], persistent=False)
        self.register_buffer('km', torch.from_numpy(__import__('numpy').stack([P54.gen_mtf(g, ratio) for g in s['ms_gnyq']])).float()[:, None], persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            dt = x.dtype; x = x.float(); c = x.shape[1]
            if c == 1: k = self.kp
            elif c == self.km.shape[0]: k = self.km
            else: raise ValueError(f'MTFDown({self.sensor}): {c} channels')
            return _blur_rep(x, k.to(x.device))[..., self.phase::self.ratio, self.phase::self.ratio].to(dt)


def apply_variant(m, variant, sensor):
    """Replace the internal resampling modules of m.core in place (no RNG consumed, no parameter added)."""
    c = m.core
    if variant == 'N': return m
    if variant not in VARIANTS: raise ValueError(variant)
    c.upsample1 = Interp23(4); c.upsample100 = Up23(1)
    if variant in ('B', 'C'):
        if sensor not in P54.SENSORS: raise ValueError('variant %s needs the training sensor, got %r' % (variant, sensor))
        c.downsample1 = MTFDown(sensor, 4, 2)
    if variant == 'C':
        c.downsample100 = MTFDown(sensor, 2, 0); c.downsample200 = MTFDown(sensor, 2, 1); c.downsample2 = MTFDown(sensor, 4, 2)
        c.upsample101 = Up23(1); c.upsample102 = Up23(0)
    return m


def Model(arm, bands=4, seed=42, sensor=None):
    sensor = sensor or SENSOR['value']
    if sensor is not None and P54.SENSORS[sensor]['bands'] != bands: raise ValueError(f'sensor {sensor} has not {bands} bands')
    if arm in FULL_SIZE: return apply_variant(to_2x2(M54.Model('C2MS', bands, seed)), arm[0], sensor)
    if arm in FACTORIZED: return apply_variant(to_2x2(M83.Model('K2', bands, seed)), arm[0], sensor)
    return M89.Model(arm, bands, seed)
