"""81 data helpers and projection variants (no side effects at import).

1. Re-exports experiment 79 prepare.py unchanged: the 54 sensor table (QB, GF2, WV3, WV2), Wald operators, and the observation-consistency
   projection of 45/66 (`Projector`, `Projected`: float64, interior margin 5, conjugate gradient per band, tolerance 1e-8, at most 400
   iterations). The official candidate (65 C2MS_L1S + P) and C2MS + P use 79 `Projected` itself, so they are the 66/79/80 projection.
2. Faster variants, fixed before evaluation, built on the same 79 Projector:
   - float32: the network output, MS and the FFT operator in float32 (complex64), tolerance 1e-5, at most 400 iterations;
   - iteration caps: float64, tolerance 1e-8, at most K iterations (K = 20, 10, 5), i.e. an early-stopped conjugate gradient.
   With float64 and the default tolerance and cap, `ProjectedVariant` runs exactly the 79 computation (checked in run.py prepare).
"""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location('exp79_prepare', Path(__file__).resolve().parent.parent / '79_deep_narrow_branch_projection/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, OBS_MARGIN = _m.SENSORS, _m.PHASE, _m.OBS_MARGIN
data_files, Wald, CoarsePairs, Projector79, Projected = _m.data_files, _m.Wald, _m.CoarsePairs, _m.Projector, _m.Projected
DTYPES = {'float64': torch.float64, 'float32': torch.float32}


def make_D(kernel, dtype, phase=PHASE, ratio=4):
    """79 make_D with the kernel spectrum in `dtype` (float32 -> complex64), so the whole projection stays in that precision."""
    k = kernel[:, 0].detach().to(dtype).cpu(); pad = k.shape[-1] // 2; spectra = {}

    def D(x):
        H, W = x.shape[-2:]
        rows = torch.arange(-pad, H + pad, device=x.device).clamp(0, H - 1); cols = torch.arange(-pad, W + pad, device=x.device).clamp(0, W - 1)
        xp = x.index_select(-2, rows).index_select(-1, cols); Hp, Wp = xp.shape[-2:]
        key = (Hp, Wp, str(x.device))
        if key not in spectra: spectra[key] = torch.fft.rfft2(torch.flip(k, (-2, -1)).to(x.device), s=(Hp, Wp))[None]
        full = torch.fft.irfft2(torch.fft.rfft2(xp, s=(Hp, Wp)) * spectra[key], s=(Hp, Wp))
        return full[..., 2 * pad:2 * pad + H, 2 * pad:2 * pad + W][..., phase::ratio, phase::ratio]
    return D


class ProjectorVariant(Projector79):
    def __init__(self, sensor, dtype='float64', tol=1e-8, max_iter=400):
        super().__init__(sensor, tol=tol, max_iter=max_iter); self.dtype = DTYPES[dtype]
        if self.dtype != torch.float64: self.D = make_D(Wald(sensor, 'cpu').ms_kernel, self.dtype)


class ProjectedVariant(torch.nn.Module):
    """79 Projected with a chosen precision, tolerance and iteration cap; returns float64 like 79."""

    def __init__(self, net, sensor, dtype='float64', tol=1e-8, max_iter=400):
        super().__init__(); self.net = net; self.projector = ProjectorVariant(sensor, dtype, tol, max_iter); self.iterations = []

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms).to(self.projector.dtype)
        with torch.inference_mode(False):
            out, it = self.projector(y.clone(), ms.to(self.projector.dtype).clone())
        self.iterations.append(it)
        return out.detach().double()
