"""Inference models of the current baseline comparison (root README, docs/c2ms_sp_result_images.md, viewer/registry.json).

original      original SSA-MRN K4 (experiment 54 C0 structure; QB weights from experiment 43, GF2/WV3 from 54)
c2ms_sp       C2MS-SP, the baseline since 2026-11-14 (experiment 81): C2MS trained with the L1 + spectral loss of experiment 65, followed by the
              observation-consistency projection (conjugate gradient stopped at 10 iterations, experiments 66/81)
c2ms_sp_lite  C2MS-SP-Lite (experiment 83): the C2MS-SP weights factorized (backbone rank 32, W2 branch 1x1-depthwise-1x1, CP rank 384) and
              fine-tuned for 10 epochs, followed by the same projection

The projection always uses the Wald operator of the sensor the image comes from. For a sensor the model was not trained on (WV2 with the
WV3 weights), C2MS-SP and C2MS-SP-Lite add the per-band sensor calibration of experiment 82 before the projection. It is never applied
to the training sensor, and never to the original SSA-MRN (the calibration and the projection are part of our method, not of the baseline). Each factory returns a module whose load_state_dict loads the plain network weights (strict), so the archived best.pt files
load unchanged. The projection runs on the CPU in float64 (MPS has no float64), the network on the module's device.
"""
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / 'experiments/performance'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_cache = {}
def _m(key):
    if key not in _cache:
        _cache[key] = {'m54': lambda: _load('bm_m54', PERF / '54_multi_sensor_scale_mixed_training/model.py'),
                       'm83': lambda: _load('bm_m83', PERF / '83_factorized_c2ms_finetuning/model.py'),
                       'p83': lambda: _load('bm_p83', PERF / '83_factorized_c2ms_finetuning/prepare.py'),
                       'p82': lambda: _load('bm_p82', PERF / '82_cross_sensor_band_calibration/prepare.py')}[key]()
    return _cache[key]


class Wrapped(nn.Module):
    """net -> optional per-band calibration (82) -> optional projection (K10); state dict = the plain network's."""

    def __init__(self, net, sensor, project=True, calibrate_from=None):
        super().__init__(); self.net = net; self.project = project
        self.calibration = None
        if calibrate_from:
            ref = json.loads((PERF / f'82_cross_sensor_band_calibration/results/reference_stats/{calibrate_from}.json').read_text(encoding='utf-8'))
            self.calibration = _m('p82').Calibrated(net, ref, 'scene')
        self.projector = _m('p83').Projector(sensor, max_iter=_m('p83').PROJECTION_ITERATIONS) if project else None

    def load_state_dict(self, state, strict=True, **kw): return self.net.load_state_dict(state, strict=strict, **kw)
    def state_dict(self, *a, **kw): return self.net.state_dict(*a, **kw)

    def forward(self, pan, lms, ms):
        y = (self.calibration or self.net)(pan, lms, ms)
        if self.projector is None: return y
        device, dtype = y.device, y.dtype
        with torch.inference_mode(False), torch.enable_grad():
            out, _ = self.projector(y.detach().cpu().double().clone(), ms.detach().cpu().double().clone())
        return out.detach().to(device=device, dtype=dtype)


def original(bands, seed, sensor, calibrate_from=None):
    return Wrapped(_m('m54').Model('C0', bands, seed), sensor, project=False, calibrate_from=calibrate_from)


def c2ms_sp(bands, seed, sensor, calibrate_from=None):
    return Wrapped(_m('m54').Model('C2MS', bands, seed), sensor, project=True, calibrate_from=calibrate_from)


def c2ms_sp_lite(bands, seed, sensor, calibrate_from=None):
    return Wrapped(_m('m83').Model('K2', bands, seed), sensor, project=True, calibrate_from=calibrate_from)


FACTORIES = {'original': original, 'c2ms_sp': c2ms_sp, 'c2ms_sp_lite': c2ms_sp_lite}
LABELS = {'original': '원본 SSA-MRN', 'c2ms_sp': 'C2MS-SP', 'c2ms_sp_lite': 'C2MS-SP-Lite'}
CHECKPOINTS = {
    'original': {'QB': 'experiments/performance/43_normalized_parallel_capacity/checkpoints/C0_QB_s{seed}/best.pt',
                 'GF2': 'experiments/performance/54_multi_sensor_scale_mixed_training/checkpoints/C0_GF2_s{seed}/best.pt',
                 'WV3': 'experiments/performance/54_multi_sensor_scale_mixed_training/checkpoints/C0_WV3_s{seed}/best.pt'},
    'c2ms_sp': {s: f'experiments/performance/65_l1_spectral_loss/checkpoints/C2MS_L1S_{s}_s{{seed}}/best.pt' for s in ('QB', 'GF2', 'WV3')},
    'c2ms_sp_lite': {s: f'experiments/performance/83_factorized_c2ms_finetuning/checkpoints/K2_{s}_s{{seed}}/best.pt' for s in ('QB', 'GF2', 'WV3')},
}
TRAINING_SENSOR = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}  # WV2 has no training data: the WV3 weights are applied (transfer)


def build(name, target, seed, calibrate=True, device='cpu'):
    """Model `name` for test sensor `target` with the archived weights of `seed`. WV2 uses the WV3 weights; C2MS-SP and C2MS-SP-Lite are then
    calibrated (experiment 82) unless calibrate=False. The original SSA-MRN is always run as published (no calibration, no projection)."""
    train = TRAINING_SENSOR[target]; bands = 8 if train in ('WV3',) else 4
    use = calibrate and train != target and name != 'original'  # the calibration is our method: never added to the original SSA-MRN
    m = FACTORIES[name](bands, seed, target, calibrate_from=train if use else None)
    ck = torch.load(ROOT / CHECKPOINTS[name][train].format(seed=seed), map_location='cpu', weights_only=False)
    if ck['seed'] != seed: raise ValueError('seed mismatch')
    m.load_state_dict(ck['model'], strict=True)
    return m.to(device).eval()
