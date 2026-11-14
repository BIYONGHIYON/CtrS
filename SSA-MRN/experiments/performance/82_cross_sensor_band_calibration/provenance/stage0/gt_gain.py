"""Post-hoc (uses test GT): oracle detail-injection relation between PAN and GT per band, per sensor.
g_b = cov(high(GT_b), high(PAN)) / var(high(PAN)); kappa_b = g_b * mean(PAN) / mean(GT_b) (relative-modulation gain, HPM units)."""
import sys, json
from pathlib import Path
import h5py, numpy as np
ROOT = Path(__file__).resolve().parents[5]
TAG = {'QB': 'qb', 'GF2': 'gf2', 'WV3': 'wv3', 'WV2': 'wv2'}
def fmask(h, w, lo, hi):
    f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]); return (f >= lo) & (f < hi)
def band(x, lo, hi): return np.real(np.fft.ifft2(np.fft.fft2(x) * fmask(*x.shape[-2:], lo, hi)))
out = {}
for s, tag in TAG.items():
    rows = []
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        for i in range(len(f['pan'])):
            pan = np.asarray(f['pan'][i], dtype=np.float64)[0]; gt = np.asarray(f['gt'][i], dtype=np.float64); lms = np.asarray(f['lms'][i], dtype=np.float64)
            r = {}
            for name, (lo, hi) in {'hf': (1 / 8, 0.51), 'b48': (1 / 8, 1 / 4), 'b24': (1 / 4, 0.51)}.items():
                ph = band(pan, lo, hi); gh = band(gt, lo, hi)
                g = (gh * ph).mean((1, 2)) / (ph ** 2).mean(); corr = (gh * ph).mean((1, 2)) / np.sqrt((gh ** 2).mean((1, 2)) * (ph ** 2).mean())
                r[name] = dict(g=g, kappa=g * pan.mean() / gt.mean((1, 2)), corr=corr, gt_over_pan_std=np.sqrt((gh ** 2).mean((1, 2)) / (ph ** 2).mean()),
                               rel_gt=np.sqrt((gh ** 2).mean((1, 2))) / gt.mean((1, 2)), rel_pan=float(np.sqrt((ph ** 2).mean()) / pan.mean()))
            r['band_over_pan'] = gt.mean((1, 2)) / pan.mean()
            r['gt_b48_over_b816'] = np.sqrt((band(gt, 1 / 8, 1 / 4) ** 2).mean((1, 2)) / (band(gt, 1 / 16, 1 / 8) ** 2).mean((1, 2)))
            r['pan_b48_over_b816'] = float(np.sqrt((band(pan, 1 / 8, 1 / 4) ** 2).mean() / (band(pan, 1 / 16, 1 / 8) ** 2).mean()))
            r['gt_b24_over_b816'] = np.sqrt((band(gt, 1 / 4, 0.51) ** 2).mean((1, 2)) / (band(gt, 1 / 16, 1 / 8) ** 2).mean((1, 2)))
            r['pan_b24_over_b816'] = float(np.sqrt((band(pan, 1 / 4, 0.51) ** 2).mean() / (band(pan, 1 / 16, 1 / 8) ** 2).mean()))
            # same-band 8-16 px relation (visible to MS): g816_b = cov(gt816_b, pan816)/var(pan816): what an input-only regression at the MS-visible band would estimate
            p8 = band(pan, 1 / 16, 1 / 8); g8 = band(gt, 1 / 16, 1 / 8); r['g816'] = (g8 * p8).mean((1, 2)) / (p8 ** 2).mean()
            l8 = band(lms, 1 / 16, 1 / 8); r['g816_lms'] = (l8 * p8).mean((1, 2)) / (p8 ** 2).mean()
            rows.append(r)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    m = lambda f: np.mean([f(r) for r in rows], 0)
    print(f'== {s}')
    print('  band/pan mean      ', m(lambda r: r['band_over_pan']))
    for name in ('hf', 'b48', 'b24'):
        print(f'  [{name}] g          ', m(lambda r: r[name]['g'])); print(f'  [{name}] kappa      ', m(lambda r: r[name]['kappa'])); print(f'  [{name}] corr       ', m(lambda r: r[name]['corr']))
        print(f'  [{name}] rel_gt     ', m(lambda r: r[name]['rel_gt']), ' rel_pan', float(m(lambda r: r[name]['rel_pan'])))
    print('  g816 (gt)          ', m(lambda r: r['g816'])); print('  g816 (lms, input)  ', m(lambda r: r['g816_lms']))
    print('  g_hf / g816_lms    ', m(lambda r: r['hf']['g'] / r['g816_lms']))
    print('  gt 4-8 / 8-16      ', m(lambda r: r['gt_b48_over_b816']), ' pan', float(m(lambda r: r['pan_b48_over_b816'])))
    print('  gt 2-4 / 8-16      ', m(lambda r: r['gt_b24_over_b816']), ' pan', float(m(lambda r: r['pan_b24_over_b816'])))
    out[s] = {k: (m(lambda r: r[k]).tolist() if not isinstance(rows[0][k], dict) else {kk: np.asarray(m(lambda r: r[k][kk])).tolist() for kk in rows[0][k]}) for k in rows[0]}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
