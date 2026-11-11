"""79 data helpers and the inference projection (no side effects at import; the 58 trainer imports this file as `prepare`).

1. Re-exports the sensor table, Wald operators and 2x coarser pairs of experiment 54 unchanged (QB entry; the 53 construction), as 77/78.
2. Observation-consistency projection of experiments 45/66, made device aware (CPU or CUDA, float64):
   y' = y + D^T M^T (M D D^T M^T)^-1 M (MS - D y), interior mask (margin 5), conjugate gradient per band (tolerance 1e-8, at most 400
   iterations). D is the 54 Wald operator (replicate pad, depthwise 41x41 cross-correlation, decimation at phase 2) in the FFT form of 66;
   D^T is taken with autograd. Projected wraps a network so that the projection runs after every inference.
"""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location('exp54_prepare', Path(__file__).resolve().parent.parent / '54_multi_sensor_scale_mixed_training/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE = _m.SENSORS, _m.PHASE, _m.SIZE
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
OBS_MARGIN = 5  # common/evaluation_protocol.OBS_MARGIN (checked in run.py prepare)


def make_D(kernel, phase=PHASE, ratio=4):
    """54 Wald operator via FFT (66 make_D), on the device of its input; the kernel spectrum is cached per padded size and device."""
    k = kernel[:, 0].detach().double().cpu(); pad = k.shape[-1] // 2; spectra = {}

    def D(x):
        H, W = x.shape[-2:]
        rows = torch.arange(-pad, H + pad, device=x.device).clamp(0, H - 1); cols = torch.arange(-pad, W + pad, device=x.device).clamp(0, W - 1)
        xp = x.index_select(-2, rows).index_select(-1, cols); Hp, Wp = xp.shape[-2:]
        key = (Hp, Wp, str(x.device))
        if key not in spectra: spectra[key] = torch.fft.rfft2(torch.flip(k, (-2, -1)).to(x.device), s=(Hp, Wp))[None]
        full = torch.fft.irfft2(torch.fft.rfft2(xp, s=(Hp, Wp)) * spectra[key], s=(Hp, Wp))
        return full[..., 2 * pad:2 * pad + H, 2 * pad:2 * pad + W][..., phase::ratio, phase::ratio]
    return D


class Projector:
    """Projection for one sensor (66 project, device aware). Must run outside torch.inference_mode (D^T uses autograd)."""

    def __init__(self, sensor, margin=OBS_MARGIN, tol=1e-8, max_iter=400):
        self.D = make_D(Wald(sensor, 'cpu').ms_kernel); self.margin, self.tol, self.max_iter = margin, tol, max_iter

    def Dt(self, v, shape):
        y = torch.zeros(shape, dtype=v.dtype, device=v.device, requires_grad=True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(self.D(y), y, grad_outputs=v)
        return g

    def __call__(self, y, ms):
        """y: (N, bands, H, W) float64 normalized output; ms: (N, bands, H/4, W/4) float64. Returns (projected y, CG iterations)."""
        m = torch.zeros_like(ms); mg = self.margin; m[..., mg:-mg, mg:-mg] = 1
        A = lambda lam: m * self.D(self.Dt(m * lam, y.shape))
        b = m * (ms - self.D(y)); lam = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum((-2, -1), keepdim=True); b2 = rs.clamp_min(1e-30); it = 0
        for it in range(1, self.max_iter + 1):
            Ap = A(p); alpha = rs / (p * Ap).sum((-2, -1), keepdim=True).clamp_min(1e-30)
            lam = lam + alpha * p; r = r - alpha * Ap; rs_new = (r * r).sum((-2, -1), keepdim=True)
            if bool(((rs_new / b2).sqrt() < self.tol).all()): break
            p = r + (rs_new / rs) * p; rs = rs_new
        return y + self.Dt(m * lam, y.shape), it


class Projected(torch.nn.Module):
    """Normalized (pan, lms, ms) in, projected normalized output out (float64); the network itself runs as usual."""

    def __init__(self, net, sensor):
        super().__init__(); self.net = net; self.projector = Projector(sensor); self.iterations = []

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms).double()
        with torch.inference_mode(False):
            out, it = self.projector(y.clone(), ms.double().clone())
        self.iterations.append(it)
        return out.detach()
