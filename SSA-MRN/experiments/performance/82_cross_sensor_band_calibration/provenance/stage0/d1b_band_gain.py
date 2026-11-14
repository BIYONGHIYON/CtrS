"""D1b (training-free, post-hoc): per-band radiometric matching of the WV2 MS to the WV3 relation with PAN.
For each WV2 scene, each MS band is mapped affinely so that its regression gain on PAN (MS scale) and/or its mean relative to PAN equal the
WV3 reference; the WV3-trained model runs in that domain and the output is mapped back. Reference = WV3 test-input averages (input-only)."""
import sys, json, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
from common.metrics import rr_metrics
torch.set_num_threads(7); torch.set_grad_enabled(False)
S = 2047.

def detail(y, gt, lms):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    high = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) * f))
    hy, hg = high(y), high(gt)
    return dict(detail_energy=float(np.sqrt((hy ** 2).mean() / (hg ** 2).mean())), injected_ratio=float((y - lms).std() / (gt - lms).std()),
                band_bias_dn=float(np.abs((y - gt).mean((1, 2))).mean()), mean_bias_dn=float((y - gt).mean()),
                detail_corr=float((hy * hg).sum() / np.sqrt((hy ** 2).sum() * (hg ** 2).sum())),
                detail_energy_band=[float(v) for v in np.sqrt((hy ** 2).mean((1, 2)) / (hg ** 2).mean((1, 2)))])

def stats(pan, ms):
    """Input-only per-band statistics on the MS grid: regression gain of MS_b on PAN_L, means."""
    pl = p66.DP(torch.from_numpy(pan)[None])[0, 0].numpy(); pc = pl - pl.mean()
    g = ((ms - ms.mean((1, 2), keepdims=True)) * pc).mean((1, 2)) / (pc ** 2).mean()
    return dict(g=g, mu=ms.mean((1, 2)), mu_p=pl.mean(), sd=ms.std((1, 2)), sd_p=pl.std())

def load(target):
    p66.use(target); tag = p66.SENSORS[target]['tag']
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        return [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]

def main(out, seeds, K, conds):
    wv3 = load('WV3'); st3 = [stats(x['pan'], x['ms']) for x in wv3]
    REF = dict(g=np.mean([s['g'] for s in st3], 0), m=np.mean([s['mu'] / s['mu_p'] for s in st3], 0), sdr=np.mean([s['sd'] / s['sd_p'] for s in st3], 0))
    print('WV3 reference', {k: np.round(v, 3).tolist() for k, v in REF.items()}, flush=True)
    wv2 = load('WV2'); st2 = [stats(x['pan'], x['ms']) for x in wv2]
    print('WV2 mean stats', dict(g=np.round(np.mean([s['g'] for s in st2], 0), 3).tolist(), m=np.round(np.mean([s['mu'] / s['mu_p'] for s in st2], 0), 3).tolist(), sdr=np.round(np.mean([s['sd'] / s['sd_p'] for s in st2], 0), 3).tolist()), flush=True)
    G2 = np.mean([s['g'] for s in st2], 0); M2 = np.mean([s['mu'] / s['mu_p'] for s in st2], 0)
    def transform(cond, x, s):
        """returns (t_b, o_b): LMS' = t*LMS + o, y = (y' - o)/t (per band)."""
        one = np.ones_like(s['g']); zero = np.zeros_like(s['g'])
        if cond == 'raw': return one, zero
        if cond == 'gain_scene': return REF['g'] / s['g'], zero                                   # scale only, per-scene regression gain
        if cond == 'gain_const': return REF['g'] / G2, zero                                       # scale only, one constant per band for the sensor
        if cond == 'mean_scene': return REF['m'] * s['mu_p'] / s['mu'], zero                      # scale only, per-scene band mean relative to PAN
        if cond == 'mean_const': return REF['m'] / M2, zero
        if cond == 'std_scene': return REF['sdr'] * s['sd_p'] / s['sd'], zero                     # scale only, per-scene std ratio (histogram-matching gain)
        if cond == 'affine_scene': t = REF['g'] / s['g']; return t, REF['m'] * s['mu_p'] - t * s['mu']   # gain by regression, mean to the WV3 mean ratio
        if cond == 'sqrt_scene': return np.sqrt((REF['g'] / s['g']) * (REF['m'] * s['mu_p'] / s['mu'])), zero
        raise ValueError(cond)
    res = {}
    for seed in seeds:
        for name, path in p66.models(seed, 'WV3').items():
            m = p66.load_model(path, 8)
            for cond in conds:
                for proj in (False, True):
                    rows = []; t0 = time.time()
                    for x, s in zip(wv2, st2):
                        t, o = transform(cond, x, s); tb, ob = t[:, None, None], o[:, None, None]
                        tin = [torch.from_numpy((a / S).astype('float32'))[None] for a in (x['pan'], tb * x['lms'] + ob, tb * x['ms'] + ob)]
                        y = m(*tin).double()[0].numpy() * S
                        y = (y - ob) / tb                                                           # back to the WV2 DN domain
                        if proj:
                            yy, _ = p66.project(torch.from_numpy(y / S)[None], torch.from_numpy(x['ms'] / S)[None], max_iter=K); y = yy[0].numpy() * S
                        r = rr_metrics(x['gt'], y); r = {k: float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')}; r.update(detail(y, x['gt'], x['lms'])); rows.append(r)
                    mean = {k: (float(np.mean([r[k] for r in rows])) if not isinstance(rows[0][k], list) else np.mean([r[k] for r in rows], 0).round(3).tolist()) for k in rows[0]}
                    key = f'{name}_s{seed}|{cond}|{"P" if proj else "raw"}'; res[key] = dict(mean=mean, rows=rows)
                    print('WV2', key, {k: (round(v, 4) if not isinstance(v, list) else v) for k, v in mean.items() if k != 'detail_energy_band'}, f'{time.time() - t0:.0f}s', flush=True)
                    Path(out).write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(',')], int(sys.argv[3]), sys.argv[4].split(','))
