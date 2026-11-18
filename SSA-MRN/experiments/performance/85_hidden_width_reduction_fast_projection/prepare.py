"""85 data helpers (the 58 trainer imports this file as `prepare`): experiment 83 prepare.py re-exported unchanged (the 79/54 sensor table,
Wald operators, 2x coarser pairs, the 45/66/79 projection and the 83 K10 wrapper `Projected`), plus `FastProjector`/`FastProjected`, the same
K10 projection computed on the interior MS grid only.

Why it is the same computation: the projection masks the outer OBS_MARGIN = 5 MS pixels. 5 MS pixels are 20 PAN pixels, the half width of the
41-tap MTF kernel (decimation phase 2), so every interior sample of D y reads real pixels only (no replicate padding) and M D is a stride-4
valid correlation starting at pixel 2. M D D^T M is then the 2-D convolution of the interior grid (RR 54x54, FR 118x118) with g = the stride-4
samples of the kernel autocorrelation (21x21, point symmetric), zero outside the interior. The conjugate gradient is the 79 loop unchanged
(float64, per image and band, tolerance 1e-8 on the relative residual of all bands, at most 10 iterations), so the result differs only by
rounding; run.py stage0 checks this on every test scene before the fast form is used. No side effects at import."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location('exp83_prepare', Path(__file__).resolve().parent.parent / '83_factorized_c2ms_finetuning/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected, PROJECTION_ITERATIONS = _m.make_D, _m.Projector, _m.Projected, _m.PROJECTION_ITERATIONS
RATIO = 4


class FastProjector:
    """Same call as 79 Projector: (y float64 (N, B, H, W), ms float64 (N, B, H/4, W/4)) -> (projected y, CG iterations). No autograd."""

    def __init__(self, sensor, device='cpu', margin=OBS_MARGIN, tol=1e-8, max_iter=PROJECTION_ITERATIONS):
        k = Wald(sensor, 'cpu').ms_kernel.double().cpu()  # (B, 1, 41, 41), the kernel of 79 make_D
        half = k.shape[-1] // 2; self.start = PHASE - half + RATIO * margin
        if self.start < 0 or PHASE - RATIO - RATIO * margin + half > -1: raise ValueError('interior samples would read padded pixels')
        R = F.conv2d(F.pad(k.transpose(0, 1), (2 * half,) * 4), k, groups=k.shape[0]).transpose(0, 1)  # autocorrelation, lags -2*half..2*half
        self.k, self.g = k.to(device), R[..., ::RATIO, ::RATIO].contiguous().to(device)
        self.gpad, self.margin, self.tol, self.max_iter, self.device = self.g.shape[-1] // 2, margin, tol, max_iter, torch.device(device)

    def D(self, y):
        lh, lw = y.shape[-2] // RATIO - 2 * self.margin, y.shape[-1] // RATIO - 2 * self.margin
        return F.conv2d(y[..., self.start:, self.start:], self.k, stride=RATIO, groups=y.shape[1])[..., :lh, :lw]

    def Dt(self, lam, shape):
        o = F.conv_transpose2d(lam, self.k, stride=RATIO, groups=lam.shape[1]); H, W = shape[-2:]
        return F.pad(o, (self.start, W - self.start - o.shape[-1], self.start, H - self.start - o.shape[-2]))

    def A(self, p): return F.conv2d(p, self.g, padding=self.gpad, groups=p.shape[1])

    def __call__(self, y, ms):
        m = self.margin; b = ms[..., m:ms.shape[-2] - m, m:ms.shape[-1] - m] - self.D(y)
        lam = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum((-2, -1), keepdim=True); b2 = rs.clamp_min(1e-30); it = 0
        for it in range(1, self.max_iter + 1):
            Ap = self.A(p); alpha = rs / (p * Ap).sum((-2, -1), keepdim=True).clamp_min(1e-30)
            lam = lam + alpha * p; r = r - alpha * Ap; rs_new = (r * r).sum((-2, -1), keepdim=True)
            if bool(((rs_new / b2).sqrt() < self.tol).all()): break
            p = r + (rs_new / rs) * p; rs = rs_new
        return y + self.Dt(lam, y.shape), it


class FastProjected(torch.nn.Module):
    """Normalized (pan, lms, ms) in, projected normalized output out (float64), like 83 Projected but with FastProjector."""

    def __init__(self, net, sensor, max_iter=PROJECTION_ITERATIONS):
        super().__init__(); self.net, self.sensor, self.max_iter = net, sensor, max_iter; self.projector = None; self.iterations = []

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms)
        with torch.autocast(device_type=ms.device.type, enabled=False):
            if self.projector is None or self.projector.device != ms.device: self.projector = FastProjector(self.sensor, ms.device, max_iter=self.max_iter)
            out, it = self.projector(y.double(), ms.double())
        self.iterations.append(it)
        return out
