"""Experiment 66 (post-hoc, no training): where the FR observation error of the L1 runs comes from, and the exact
observation-consistency projection of experiment 45 applied at inference. QB, per seed, local CPU.

Models (archived best.pt, --seed): 53 C2MS, 63 C2MS L1 (seed 42 only), 65 C2MS L1 + spectral loss. Metrics follow the server evaluation of
53/54/63/65 (RR, RR2, FR_WALD, FR + observation RMSE with margin 5) for (a) the raw output and (b) the output after
y' = y + D^T M^T (M D D^T M^T)^-1 M (MS - D y) (interior mask, margin 5, CG in float64) applied at every inference.
FR observation error e = D(y) - MS (interior) is split into the per-band mean bias, the part from the output's band
below the MS Nyquist frequency (ideal FFT lowpass at 1/8 cycles/pixel) and the part from the band above it (aliasing).
The 54 Wald operators are used in an FFT form (identical to the direct depthwise form within 3e-15, checked here),
because the direct 41x41 float64 convolution is very slow on CPU. Writes only below results/.

--sensor GF2/WV3 (65 stage 3): 54 C2MS and 65 L1 + spectral of the same seed, on the sensor's own test set and, for WV3,
also on WV2 (transfer), each with the target's Wald operator and normalization (GF2 1023, WV3/WV2 2047) as in the 54/65
evaluation; v2.3 judgements of L1S, L1S+P and C2MS+P against C2MS (and L1S+P against C2MS+P) are added per target.

--median-test (QB): whether the extra FR observation error of the L1 runs comes from L1 estimating the conditional median
instead of the mean (see median_test).
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; PERF = ROOT / 'experiments/performance'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PERF / '65_l1_spectral_loss'))
from model import Model  # noqa: E402  (65 model.py: 54 C2MS)
from prepare import Wald, SENSORS  # noqa: E402  (65 prepare.py: 54 sensor operators)
from common.metrics import rr_metrics, fr_metrics  # noqa: E402
from common.evaluation_protocol import OBS_MARGIN  # noqa: E402
from common.interpolation import interp23tap  # noqa: E402

torch.set_num_threads(8); torch.set_grad_enabled(False)
S = 2047.
TARGETS = {'GF2': ('GF2',), 'WV3': ('WV3', 'WV2')}  # training sensor -> test targets (WV2 = transfer of the WV3 models)


def models(seed, sensor='QB'):
    """Archived best.pt per seed; 63 (L1 only) exists for seed 42 only. GF2/WV3: 54 C2MS and 65 stage-3 L1S."""
    if sensor != 'QB':
        return {'C2MS': PERF / f'54_multi_sensor_scale_mixed_training/checkpoints/C2MS_{sensor}_s{seed}/best.pt',
                'L1S': PERF / f'65_l1_spectral_loss/checkpoints/C2MS_L1S_{sensor}_s{seed}/best.pt'}
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


def use(target):
    """Switch the operators and normalization used by every function below to the test target's sensor."""
    global W, D, DP, S
    W = Wald(target, 'cpu'); D = make_D(W.ms_kernel); DP = make_D(W.pan_kernel); S = W.max_value


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


def load_model(path, bands=4):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = Model('C2MS', bands, int(ck['seed'])); m.load_state_dict(ck['model'], strict=True); return m.eval()


class Net:
    """DN arrays in, DN array out; optionally projected."""

    def __init__(self, path, projected, bands=4):
        self.m = load_model(path, bands); self.projected = projected; self.iters = []

    def __call__(self, pan, lms, ms):
        x = [torch.from_numpy((np.asarray(a, dtype=np.float64) / S).astype('float32'))[None] for a in (pan, lms, ms)]
        y = self.m(*x).double()
        if self.projected:
            y, it = project(y, x[2].double()); self.iters.append(it)
        return y[0].numpy() * S


class Projected(torch.nn.Module):
    """Normalized tensors in/out (common/result_images interface), projection at inference."""

    def __init__(self, path, bands=4):
        super().__init__(); self.m = load_model(path, bands)

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
    x = torch.rand(1, W.ms_kernel.shape[0], 96, 96, dtype=torch.float64); p = torch.rand(1, 1, 96, 96, dtype=torch.float64)
    return dict(ms_max_abs=float((W.ms(x) - D(x)).abs().max()), pan_max_abs=float((W.pan(p) - DP(p)).abs().max()))


def plot(res, out, seed, stem=None, label=None):
    stem = stem or f'analysis_s{seed}'; label = label or f'seed {seed}'
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = tuple(n for n in ('C2MS', 'L1', 'L1S') if n in res['models']); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.arange(len(names)); w = 0.25
    for i, k in enumerate(('bias', 'low_band', 'high_band_aliasing')):
        ax[0].bar(x + (i - 1) * w, [res['models'][n]['mean']['obs_decomposition'][k] for n in names], w, label=k)
    ax[0].set_xticks(x, names); ax[0].set_ylabel('RMSE (DN)'); ax[0].set_title(f'FR observation error split ({label}, 20 scenes)'); ax[0].legend(fontsize=8)
    base = res['models']['C2MS']['mean']; keys = [n + s for n in names for s in ('', '+P')]
    for i, part in enumerate(('RR', 'RR2', 'FR_WALD')):
        ax[1].bar(np.arange(len(keys)) + (i - 1) * w, [res['models'][k]['mean'][part]['PSNR'] - base[part]['PSNR'] for k in keys], w, label=part)
    ax[1].axhline(0, c='k', lw=.8); ax[1].set_xticks(range(len(keys)), keys, fontsize=8); ax[1].set_title('PSNR - C2MS (dB); +P = with projection'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f'{stem}.png', dpi=120); plt.close(fig)


def run_sensor(a):
    """GF2/WV3 (+WV2): per test target, C2MS and L1S with and without projection, the error split and v2.3 judgements."""
    spec = importlib.util.spec_from_file_location('c2ms54_eval', PERF / '54_multi_sensor_scale_mixed_training/evaluate.py')
    V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)  # judge(), sha() and the expected test-data hashes
    MODELS = models(a.seed, a.sensor)
    if a.c2ms: MODELS['C2MS'] = Path(a.c2ms)
    if a.l1s: MODELS['L1S'] = Path(a.l1s)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = f'analysis_{a.sensor}_s{a.seed}'; target_file = out / f'{stem}.json'
    if a.plot_only:
        res = json.loads(target_file.read_text(encoding='utf-8'))
        for t, tr in res['targets'].items(): plot(tr, out, a.seed, f'{stem}_{t}', f'{t}, seed {a.seed}')
        return
    data = Path(a.data) if a.data else ROOT / 'data/test'; bands = SENSORS[a.sensor]['bands']; t0 = time.time()
    if a.images_only: return sensor_images(a, MODELS, data)
    res = {'seed': a.seed, 'sensor': a.sensor, 'targets': {}}
    for t in TARGETS[a.sensor]:
        use(t); paths = {}
        for proto, name in V.files(t).items():
            found = list(data.rglob(name)); assert len(found) == 1, name; paths[proto] = found[0]
            if V.sha(found[0]) != V.expected_sha(t, proto): raise ValueError(f'test data sha mismatch {t} {proto}')
        with h5py.File(paths['RR']) as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
        with h5py.File(paths['FR']) as f: fr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')} for i in range(20)]
        tr = res['targets'][t] = {'operator_check': operator_check(), 'models': {}}
        for name, path in MODELS.items():
            for projected in (False, True):
                net = Net(path, projected, bands); key = name + ('+P' if projected else ''); rows = [scene(net, x, y) for x, y in zip(rr, fr)]
                mean = {p: {k: float(np.mean([r[p][k] for r in rows])) for k in rows[0][p] if not isinstance(rows[0][p][k], list)} for p in rows[0] if isinstance(rows[0][p], dict)}
                tr['models'][key] = dict(checkpoint=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), mean=mean, rows=rows, cg_iterations_max=max(net.iters) if net.iters else None)
                print(t, key, {p: {k: round(v, 4) for k, v in mean[p].items()} for p in mean}, round(time.time() - t0), 's', flush=True)
                target_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        m = {k: {p: v['mean'][p] for p in ('RR', 'RR2', 'FR_WALD', 'FR')} for k, v in tr['models'].items()}
        tr['judgement'] = {}
        for cand, ctrl in (('L1S', 'C2MS'), ('L1S+P', 'C2MS'), ('C2MS+P', 'C2MS'), ('L1S+P', 'C2MS+P')):
            if m[ctrl]['FR']['OBS_RMSE_DN'] > 1e-9: tr['judgement'][f'{cand} vs {ctrl}'] = V.judge([m[cand]], [m[ctrl]])  # no observation ratio against a projected control
        print(t, {k: v['label'] for k, v in tr['judgement'].items()}, flush=True)
        target_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); plot(tr, out, a.seed, f'{stem}_{t}', f'{t}, seed {a.seed}')
    res['seconds'] = round(time.time() - t0, 1); target_file.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    if a.seed == 42: sensor_images(a, MODELS, data)


def sensor_images(a, MODELS, data):
    """Fixed six RR scenes (same indices as QB), seed 42: 65 L1S without vs with projection, per test target."""
    if a.seed != 42: return  # result images use the fixed seed 42 only
    from common import result_images
    bands = SENSORS[a.sensor]['bands']
    for t in TARGETS[a.sensor]:
        use(t); s = SENSORS[t]
        result_images.render(load_model(MODELS['L1S'], bands), Projected(MODELS['L1S'], bands), data, Path(a.out) / 'images' / f'L1S_vs_L1S_projected_{t}',
                             names=('L1S', 'L1S+P'), seed=42, device='cpu', labels=('65번 C2MS L1 + 분광', '65번 + 관측 일치 사영'),
                             rr_file=f"test_{s['tag']}_multiExm1.h5", scale=s['max_value'], bands=s['rgb'], land_cover={i: f'{t} 장면 {i}' for i in result_images.SCENES})


def skew_proxy(fine, h, w):
    """(mean - median) / std and std over the 8x8 fine window around each MS sample (phase 2: centre 4i+2), per band."""
    x = torch.nn.functional.pad(torch.from_numpy(fine)[None], (2, 4, 2, 4), mode='replicate')  # window rows 4i-2 .. 4i+5
    u = x.unfold(2, 8, 4).unfold(3, 8, 4)[0, :, :h, :w].reshape(fine.shape[0], h, w, 64)
    std = u.std(-1)
    return ((u.mean(-1) - u.median(-1).values) / std.clamp_min(1e-6)).numpy(), std.numpy()


def median_test(rr_path, fr_path, out):
    """Mean-vs-median test (QB, seeds 42-44; 63 L1 seed 42): MSE learns the conditional mean, L1 the conditional median.
    D is linear, so a mean estimate keeps D(y) = MS. Per MS pixel (interior), d = e_A - e_B with e = D(y) - MS:
    (1) local test: correlation of d with the fine-scale (mean - median) around the sample (RR: GT, FR: PAN), MSE-MSE pairs as control;
    (2) global test: mean of d overall, by MS-brightness and PAN-std quintile and per band; skew of the GT - LMS residual on RR."""
    use('QB'); k = OBS_MARGIN; sl = (slice(None), slice(k, -k), slice(k, -k))
    names = {f'C2MS_s{s}': models(s)['C2MS'] for s in (42, 43, 44)} | {'L1_s42': models(42)['L1']} | {f'L1S_s{s}': models(s)['L1S'] for s in (42, 43, 44)}
    nets = {n: Net(p, False) for n, p in names.items()}
    pairs = [('L1_s42', 'C2MS_s42'), ('L1S_s42', 'C2MS_s42'), ('L1S_s43', 'C2MS_s43'), ('L1S_s44', 'C2MS_s44'), ('C2MS_s43', 'C2MS_s42'), ('C2MS_s44', 'C2MS_s43')]
    q5 = lambda v: np.quantile(v, np.linspace(0, 1, 6)); res = {'models': {n: str(p.relative_to(ROOT)) for n, p in names.items()}, 'pairs_control': pairs[-2:]}
    for scale, path in (('RR', rr_path), ('FR', fr_path)):
        E = {n: [] for n in nets}; SK, SD, BR, BAND, RES = [], [], [], [], []
        with h5py.File(path) as f:
            for i in range(20):
                a = {key: np.asarray(f[key][i], dtype=np.float64) for key in ('pan', 'lms', 'ms') + (('gt',) if scale == 'RR' else ())}
                h, w = a['ms'].shape[-2:]
                sk, sd = skew_proxy(a['gt'] if scale == 'RR' else np.repeat(a['pan'], 4, 0), h, w)
                SK.append(sk[sl].ravel()); SD.append(sd[sl].ravel()); BR.append(a['ms'][sl].ravel())
                BAND.append(np.broadcast_to(np.arange(4)[:, None, None], a['ms'][sl].shape).ravel())
                if scale == 'RR': RES.append((a['gt'] - a['lms']).ravel())
                for n, net in nets.items():
                    E[n].append((D(torch.from_numpy(net(a['pan'], a['lms'], a['ms']))[None])[0].numpy() - a['ms'])[sl].ravel())
        sk, sd, br, band = map(np.concatenate, (SK, SD, BR, BAND)); E = {n: np.concatenate(v) for n, v in E.items()}
        r = {'obs_rmse_dn': {n: float(np.sqrt((v ** 2).mean())) for n, v in E.items()}, 'pairs': {}}
        if RES:
            v = np.concatenate(RES); r['gt_minus_lms'] = dict(mean=float(v.mean()), median=float(np.median(v)), std=float(v.std()), skew=float(((v - v.mean()) ** 3).mean() / v.std() ** 3))
        qs, qb, qk = q5(sd), q5(br), np.quantile(sk, np.linspace(0, 1, 11))
        by = lambda d, x, q: [float(d[(x >= q[j]) & (x <= q[j + 1])].mean()) for j in range(len(q) - 1)]
        for A, B in pairs:
            d = E[A] - E[B]; x = sk * sd
            r['pairs'][f'{A}-{B}'] = dict(mean_dn=float(d.mean()), rms_dn=float(np.sqrt((d ** 2).mean())), corr_mean_minus_median=float(np.corrcoef(d, x)[0, 1]),
                                          corr_skew=float(np.corrcoef(d, sk)[0, 1]), mean_by_skew_decile=by(d, sk, qk), mean_by_brightness_quintile=by(d, br, qb),
                                          mean_by_pan_std_quintile=by(d, sd, qs), rms_by_pan_std_quintile=[float(np.sqrt((d[(sd >= qs[j]) & (sd <= qs[j + 1])] ** 2).mean())) for j in range(5)],
                                          mean_by_band=[float(d[band == c].mean()) for c in range(4)], corr_abs_pan_std=float(np.corrcoef(np.abs(d), sd)[0, 1]))
            p = r['pairs'][f'{A}-{B}']; print(scale, A, B, {key: (round(v, 3) if isinstance(v, float) else [round(z, 2) for z in v]) for key, v in p.items()}, flush=True)
        res[scale] = r
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    a = argparse.ArgumentParser(); a.add_argument('--seed', type=int, default=42); a.add_argument('--plot-only', action='store_true'); a.add_argument('--images-only', action='store_true')
    # Server run (2026-10-27): the same QB test files from another folder, the 65 best.pt from its runtime, results outside the repository.
    a.add_argument('--rr', default=str(ROOT / 'data/test/test_qb_multiExm1.h5')); a.add_argument('--fr', default=str(ROOT / 'data/test/test_qb_OrigScale_multiExm1.h5')); a.add_argument('--l1s'); a.add_argument('--out', default=str(HERE / 'results'))
    # GF2/WV3 (+WV2 transfer): --data is searched for the test files by name (default data/test), --c2ms/--l1s override the archived best.pt.
    a.add_argument('--sensor', default='QB', choices=('QB', 'GF2', 'WV3')); a.add_argument('--data'); a.add_argument('--c2ms'); a.add_argument('--threads', type=int, default=8)
    a.add_argument('--median-test', action='store_true', help='QB mean-vs-median test of the L1 observation error -> results/median_vs_mean.json'); a = a.parse_args()
    torch.set_num_threads(a.threads)
    if a.median_test: return median_test(Path(a.rr), Path(a.fr), Path(a.out) / 'median_vs_mean.json')
    if a.sensor != 'QB': return run_sensor(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); target = out / f'analysis_s{a.seed}.json'; MODELS = models(a.seed)
    if a.l1s: MODELS['L1S'] = Path(a.l1s)
    if a.plot_only: plot(json.loads(target.read_text(encoding='utf-8')), out, a.seed); return
    if not a.images_only:
        t0 = time.time(); res = {'seed': a.seed, 'operator_check': operator_check(), 'models': {}}
        with h5py.File(a.rr) as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
        with h5py.File(a.fr) as f: fr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')} for i in range(20)]
        for name, path in MODELS.items():
            for projected in (False, True):
                net = Net(path, projected); key = name + ('+P' if projected else ''); rows = [scene(net, x, y) for x, y in zip(rr, fr)]
                mean = {p: {k: float(np.mean([r[p][k] for r in rows])) for k in rows[0][p] if not isinstance(rows[0][p][k], list)} for p in rows[0] if isinstance(rows[0][p], dict)}
                res['models'][key] = dict(checkpoint=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), mean=mean, rows=rows, cg_iterations_max=max(net.iters) if net.iters else None)
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
