"""Experiment 66 (post-hoc, no training): where the FR observation error of the L1 runs comes from, and the exact
observation-consistency projection of experiment 45 applied at inference. QB, per seed, local CPU.

Models (archived best.pt, --seed): 53 C2MS, 63 C2MS L1 (seed 42 only), 65 C2MS L1 + spectral loss. Metrics follow the server evaluation of
53/54/63/65 (RR, RR2, FR_WALD, FR + observation RMSE with margin 5) for (a) the raw output and (b) the output after
y' = y + D^T M^T (M D D^T M^T)^-1 M (MS - D y) (interior mask, margin 5, CG in float64) applied at every inference.
FR observation error e = D(y) - MS (interior) is split into the per-band mean bias, the part from the output's band
below the MS Nyquist frequency (ideal FFT lowpass at 1/8 cycles/pixel) and the part from the band above it (aliasing).
The 54 Wald operators are used in an FFT form (identical to the direct depthwise form within 3e-15, checked here),
because the direct 41x41 float64 convolution is very slow on CPU. Writes only below results/.
"""
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; PERF = ROOT / 'experiments/performance'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PERF / '65_l1_spectral_loss'))
from model import Model  # noqa: E402  (65 model.py: 54 C2MS)
from prepare import Wald  # noqa: E402  (65 prepare.py: 54 sensor operators)
from common.metrics import rr_metrics, fr_metrics  # noqa: E402
from common.evaluation_protocol import OBS_MARGIN  # noqa: E402
from common.interpolation import interp23tap  # noqa: E402

torch.set_num_threads(8); torch.set_grad_enabled(False)
S = 2047.
def models(seed):
    """Archived best.pt per seed; 63 (L1 only) exists for seed 42 only."""
    m = {'C2MS': PERF / f'53_scale_mixed_training/checkpoints/C2MS_QB_s{seed}/best.pt',
         'L1': PERF / f'63_l1_reconstruction_loss/checkpoints/C2MS_L1_QB_s{seed}/best.pt',
         'L1S': PERF / f'65_l1_spectral_loss/checkpoints/C2MS_L1S_QB_s{seed}/best.pt'}
    return {k: v for k, v in m.items() if v.exists()}
W = Wald('QB', 'cpu')


def make_D(kernel, phase=2, ratio=4):
    """54 Wald operator (replicate pad, depthwise cross-correlation, decimation at phase 2) via FFT."""
    k = kernel[:, 0].double(); pad = k.shape[-1] // 2

    def D(x):
        H, Wd = x.shape[-2:]
        rows = torch.arange(-pad, H + pad).clamp(0, H - 1); cols = torch.arange(-pad, Wd + pad).clamp(0, Wd - 1)
        xp = x.index_select(-2, rows).index_select(-1, cols); Hp, Wp = xp.shape[-2:]
        full = torch.fft.irfft2(torch.fft.rfft2(xp, s=(Hp, Wp)) * torch.fft.rfft2(torch.flip(k, (-2, -1)), s=(Hp, Wp))[None], s=(Hp, Wp))
        return full[..., 2 * pad:2 * pad + H, 2 * pad:2 * pad + Wd][..., phase::ratio, phase::ratio]
    return D


D = make_D(W.ms_kernel); DP = make_D(W.pan_kernel)


def Dt(v, shape):
    y = torch.zeros(shape, dtype=v.dtype, requires_grad=True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(D(y), y, grad_outputs=v)
    return g


def project(y, ms, margin=OBS_MARGIN, tol=1e-8, max_iter=400):
    """Experiment 45 projection: minimum-norm correction so that the interior observation of y equals ms (CG per band)."""
    m = torch.zeros_like(ms); m[..., margin:-margin, margin:-margin] = 1
    A = lambda lam: m * D(Dt(m * lam, y.shape))
    b = m * (ms - D(y)); lam = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r * r).sum((-2, -1), keepdim=True); b2 = rs.clamp_min(1e-30); it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p); alpha = rs / (p * Ap).sum((-2, -1), keepdim=True).clamp_min(1e-30)
        lam = lam + alpha * p; r = r - alpha * Ap; rs_new = (r * r).sum((-2, -1), keepdim=True)
        if bool(((rs_new / b2).sqrt() < tol).all()): break
        p = r + (rs_new / rs) * p; rs = rs_new
    return y + Dt(m * lam, y.shape), it


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = Model('C2MS', 4, int(ck['seed'])); m.load_state_dict(ck['model'], strict=True); return m.eval()


class Net:
    """DN arrays in, DN array out; optionally projected."""

    def __init__(self, path, projected):
        self.m = load_model(path); self.projected = projected; self.iters = []

    def __call__(self, pan, lms, ms):
        x = [torch.from_numpy((np.asarray(a, dtype=np.float64) / S).astype('float32'))[None] for a in (pan, lms, ms)]
        y = self.m(*x).double()
        if self.projected:
            y, it = project(y, x[2].double()); self.iters.append(it)
        return y[0].numpy() * S


class Projected(torch.nn.Module):
    """Normalized tensors in/out (common/result_images interface), projection at inference."""

    def __init__(self, path):
        super().__init__(); self.m = load_model(path)

    def forward(self, pan, lms, ms):
        return project(self.m(pan, lms, ms).double(), ms.double())[0]


def obs_rmse(y, ms, margin=OBS_MARGIN):
    e = D(torch.from_numpy(np.asarray(y, dtype=np.float64))[None])[0].numpy() - ms
    return float(np.sqrt(np.mean(e[..., margin:-margin, margin:-margin] ** 2)))


def degrade(pan, ms):
    pb = DP(torch.from_numpy(pan)[None])[0].numpy(); mb = D(torch.from_numpy(ms)[None])
    return pb, (interp23tap((mb / S).float()).double() * S)[0].numpy(), mb[0].numpy()


def lowpass(y, cut=1 / 8):
    h, w = y.shape[-2:]; fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(y) * ((np.abs(fy) <= cut) & (np.abs(fx) <= cut))))


def decompose(y, ms):
    t = lambda a: torch.from_numpy(a)[None]; k = OBS_MARGIN; inner = lambda e: e[..., k:-k, k:-k]
    e = inner(D(t(y))[0].numpy() - ms); yl = lowpass(y)
    e_lf = inner(D(t(yl))[0].numpy() - ms); e_hf = inner(D(t(y - yl))[0].numpy())
    bias = e.mean(axis=(1, 2), keepdims=True); rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    return dict(total=rms(e), bias=rms(np.broadcast_to(bias, e.shape)), without_bias=rms(e - bias), low_band=rms(e_lf),
                high_band_aliasing=rms(e_hf), band_bias=[float(x) for x in bias.ravel()])


def scene(net, rr, fr):
    out = {'RR': rr_metrics(rr['gt'], net(rr['pan'], rr['lms'], rr['ms'])),
           'RR2': rr_metrics(rr['ms'], net(*degrade(rr['pan'], rr['ms']))),
           'FR_WALD': rr_metrics(fr['ms'], net(*degrade(fr['pan'], fr['ms'])))}
    y = net(fr['pan'], fr['lms'], fr['ms']); out['FR'] = fr_metrics(y, fr['lms'], fr['ms'], fr['pan']); out['FR']['OBS_RMSE_DN'] = obs_rmse(y, fr['ms'])
    if not net.projected: out['obs_decomposition'] = decompose(y, fr['ms'])
    else: out['projection_change_dn'] = None
    return out


def operator_check():
    x = torch.rand(1, 4, 96, 96, dtype=torch.float64); p = torch.rand(1, 1, 96, 96, dtype=torch.float64)
    return dict(ms_max_abs=float((W.ms(x) - D(x)).abs().max()), pan_max_abs=float((W.pan(p) - DP(p)).abs().max()))


def plot(res, out, seed):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = tuple(n for n in ('C2MS', 'L1', 'L1S') if n in res['models']); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.arange(len(names)); w = 0.25
    for i, k in enumerate(('bias', 'low_band', 'high_band_aliasing')):
        ax[0].bar(x + (i - 1) * w, [res['models'][n]['mean']['obs_decomposition'][k] for n in names], w, label=k)
    ax[0].set_xticks(x, names); ax[0].set_ylabel('RMSE (DN)'); ax[0].set_title(f'FR observation error split (seed {seed}, 20 scenes)'); ax[0].legend(fontsize=8)
    base = res['models']['C2MS']['mean']; keys = [n + s for n in names for s in ('', '+P')]
    for i, part in enumerate(('RR', 'RR2', 'FR_WALD')):
        ax[1].bar(np.arange(len(keys)) + (i - 1) * w, [res['models'][k]['mean'][part]['PSNR'] - base[part]['PSNR'] for k in keys], w, label=part)
    ax[1].axhline(0, c='k', lw=.8); ax[1].set_xticks(range(len(keys)), keys, fontsize=8); ax[1].set_title('PSNR - C2MS (dB); +P = with projection'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f'analysis_s{seed}.png', dpi=120); plt.close(fig)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--seed', type=int, default=42); a.add_argument('--plot-only', action='store_true'); a.add_argument('--images-only', action='store_true'); a = a.parse_args()
    out = HERE / 'results'; out.mkdir(exist_ok=True); target = out / f'analysis_s{a.seed}.json'; MODELS = models(a.seed)
    if a.plot_only: plot(json.loads(target.read_text(encoding='utf-8')), out, a.seed); return
    if not a.images_only:
        t0 = time.time(); res = {'seed': a.seed, 'operator_check': operator_check(), 'models': {}}
        with h5py.File(ROOT / 'data/test/test_qb_multiExm1.h5') as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
        with h5py.File(ROOT / 'data/test/test_qb_OrigScale_multiExm1.h5') as f: fr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')} for i in range(20)]
        for name, path in MODELS.items():
            for projected in (False, True):
                net = Net(path, projected); key = name + ('+P' if projected else ''); rows = [scene(net, x, y) for x, y in zip(rr, fr)]
                mean = {p: {k: float(np.mean([r[p][k] for r in rows])) for k in rows[0][p] if not isinstance(rows[0][p][k], list)} for p in rows[0] if isinstance(rows[0][p], dict)}
                res['models'][key] = dict(checkpoint=str(path.relative_to(ROOT)), mean=mean, rows=rows, cg_iterations_max=max(net.iters) if net.iters else None)
                print(key, {p: {k: round(v, 4) for k, v in mean[p].items()} for p in mean}, round(time.time() - t0), 's', flush=True)
                target.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        res['seconds'] = round(time.time() - t0, 1); target.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        plot(res, out, a.seed)
    if a.seed != 42: return  # result images use the fixed seed 42 only
    from common import result_images
    result_images.render(load_model(MODELS['L1S']), Projected(MODELS['L1S']), ROOT / 'data', out / 'images' / 'L1S_vs_L1S_projected', names=('L1S', 'L1S+P'), seed=42, device='cpu',
                         labels=('65번 C2MS L1 + 분광', '65번 + 관측 일치 사영'))


if __name__ == '__main__':
    main()
