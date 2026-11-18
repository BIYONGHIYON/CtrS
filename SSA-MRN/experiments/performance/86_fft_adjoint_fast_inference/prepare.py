"""86 inference helpers: experiment 85 prepare.py re-exported unchanged (the 79/54 sensor table, Wald operators, the 79/83 K10 projection
`Projector`/`Projected` and the 85 interior-grid `FastProjector`/`FastProjected`), plus:

- FastProjectorV2: the 85 FastProjector with the last step D^T computed by FFT instead of the float64 stride-4 transposed convolution, whose
  cuDNN algorithm without autotune made 85 slow at 512^2 (QB 388 ms for that one call). D^T lambda is the full convolution of the interior
  samples placed on the PAN grid (positions start + 4 j) with the 41-tap kernel; with an FFT size of at least H + 40 there is no wrap-around,
  so the result is the same up to rounding. D (strided correlation) and the 21x21 interior operator stay as in 85.
- Graphed: replays a network through a CUDA graph (static input shapes), used only for the inference-setting timings.
No side effects at import."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location('exp85_prepare', Path(__file__).resolve().parent.parent / '85_hidden_width_reduction_fast_projection/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN, RATIO = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN, _m.RATIO
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected, PROJECTION_ITERATIONS = _m.make_D, _m.Projector, _m.Projected, _m.PROJECTION_ITERATIONS
FastProjector, FastProjected = _m.FastProjector, _m.FastProjected


def _fft_size(n):
    """Smallest size >= n whose only prime factors are 2, 3 and 5 (fast cuFFT sizes)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0: m //= p
        if m == 1: return n
        n += 1


class FastProjectorV2(FastProjector):
    """FastProjector with D^T by FFT; same call and stop rule."""

    def __init__(self, sensor, device='cpu', **kw):
        super().__init__(sensor, device, **kw); self.spectra = {}

    def Dt(self, lam, shape):
        H, W = shape[-2:]; kh, kw = self.k.shape[-2:]; sh, sw = _fft_size(H + kh - 1), _fft_size(W + kw - 1)
        up = torch.zeros(lam.shape[:2] + (H, W), dtype=lam.dtype, device=lam.device)
        up[..., self.start:self.start + RATIO * lam.shape[-2]:RATIO, self.start:self.start + RATIO * lam.shape[-1]:RATIO] = lam
        key = (sh, sw, str(lam.device), lam.dtype)
        if key not in self.spectra: self.spectra[key] = torch.fft.rfft2(self.k[:, 0].to(lam.dtype), s=(sh, sw))[None]
        return torch.fft.irfft2(torch.fft.rfft2(up, s=(sh, sw)) * self.spectra[key], s=(sh, sw))[..., :H, :W]


class FastProjectedV2(FastProjected):
    """85 FastProjected with FastProjectorV2."""

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms)
        with torch.autocast(device_type=ms.device.type, enabled=False):
            if self.projector is None or self.projector.device != ms.device: self.projector = FastProjectorV2(self.sensor, ms.device, max_iter=self.max_iter)
            out, it = self.projector(y.double(), ms.double())
        self.iterations.append(it)
        return out


class Graphed(torch.nn.Module):
    """Network replayed through a CUDA graph per input shape (timing only; outputs are copies of the static output)."""

    def __init__(self, net):
        super().__init__(); self.net = net; self.graphs = {}

    def forward(self, pan, lms, ms):
        key = tuple(tuple(x.shape) for x in (pan, lms, ms))
        if key not in self.graphs:
            static = [x.clone() for x in (pan, lms, ms)]; side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3): self.net(*static)
            torch.cuda.current_stream().wait_stream(side); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = self.net(*static)
            self.graphs[key] = (g, static, out)
        g, static, out = self.graphs[key]
        for s, x in zip(static, (pan, lms, ms)): s.copy_(x)
        g.replay(); return out.clone()
