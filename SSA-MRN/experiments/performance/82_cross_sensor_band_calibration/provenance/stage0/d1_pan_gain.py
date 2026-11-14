"""D1 (training-free, post-hoc): does matching the WV2 PAN radiometry to the WV3 training relation remove the WV2 transfer blur/bias?
WV3-trained 54 C2MS and 65 C2MS_L1S (seeds 42-44) on WV2 (transfer) and WV3 RR test, PAN rescaled by input-only statistics."""
import sys, json, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
from common.metrics import rr_metrics
torch.set_num_threads(6); torch.set_grad_enabled(False)
S = 2047.
REF = dict(ratio=1.1641, slope=1.1455, offset=3.35, std_ratio=1.1781)   # WV3: training mean ratio (67 panratio.json); slope/offset/std ratio from WV3 test inputs

def detail(y, gt, lms):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    high = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) * f))
    return dict(detail_energy=float(np.sqrt((high(y) ** 2).mean() / (high(gt) ** 2).mean())), injected_ratio=float((y - lms).std() / (gt - lms).std()),
                band_bias_dn=float(np.abs((y - gt).mean((1, 2))).mean()), mean_bias_dn=float((y - gt).mean()))

def pan_variants(pan, lms, ms, target):
    I = ms.mean(0); pl = p66.DP(torch.from_numpy(pan)[None])[0, 0].numpy()          # PAN on the MS grid with the test sensor's PAN MTF
    A = np.stack([I.ravel(), np.ones(I.size)], 1); (a, b), *_ = np.linalg.lstsq(A, pl.ravel(), rcond=None)
    r = pan.mean() / lms.mean()
    v = {'raw': pan,
         'mean_ratio': pan * (REF['ratio'] / r),                                         # per-scene scalar gain to the WV3 training mean ratio
         'affine_reg': (pan - b) / a * REF['slope'] + REF['offset'],                     # per-scene regression PAN_L ~ a I + b mapped to the WV3 relation
         'mean_std': (pan - pan.mean()) * (REF['std_ratio'] * I.std() / pl.std()) + REF['ratio'] * lms.mean()}  # per-scene mean/std matching
    if target == 'WV2': v['const_gain'] = pan * (REF['ratio'] / 0.8762)                  # one sensor-level constant (WV2 test mean ratio 0.8762)
    return v

def main(out, seeds, K):
    res = {}
    for target in ('WV2', 'WV3'):
        p66.use(target); tag = p66.SENSORS[target]['tag']
        with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
            rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
        variants = [pan_variants(x['pan'], x['lms'], x['ms'], target) for x in rr]
        res[target] = {}
        for seed in seeds:
            for name, path in p66.models(seed, 'WV3').items():
                m = p66.load_model(path, 8)
                for cond in variants[0]:
                    for proj in (False, True):
                        rows = []; t0 = time.time()
                        for x, v in zip(rr, variants):
                            t = [torch.from_numpy((a / S).astype('float32'))[None] for a in (v[cond], x['lms'], x['ms'])]
                            y = m(*t).double()
                            if proj: y, _ = p66.project(y, t[2].double(), max_iter=K)
                            y = y[0].numpy() * S
                            r = rr_metrics(x['gt'], y); r = {k: float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')}; r.update(detail(y, x['gt'], x['lms'])); rows.append(r)
                        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                        key = f'{name}_s{seed}|{cond}|{"P" if proj else "raw"}'; res[target][key] = dict(mean=mean, rows=rows)
                        print(target, key, {k: round(v, 4) for k, v in mean.items()}, f'{time.time() - t0:.0f}s', flush=True)
                        Path(out).write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(',')], int(sys.argv[3]))
