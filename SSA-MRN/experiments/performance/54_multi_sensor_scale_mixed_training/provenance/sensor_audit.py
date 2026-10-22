"""Sensor audit (no training): which x4 MS operator reproduces each PanCollection dataset's MS from its GT.

For every dataset: array shapes, counts, value ranges; for MS = decimate(MTF(GT)) the relative RMSE against the given MS
for the DLPan-Toolbox GNyq (if sourced), uniform GNyq candidates and per-band best GNyq, at sampling phases 0-3; and
whether interp23tap(MS) reproduces the given LMS. Training splits are used where they exist; WV2 has only test data.
"""
import sys, json
from pathlib import Path
import h5py, numpy as np, torch
from torch.nn import functional as F

ROOT, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(ROOT))
from common.interpolation import interp23tap  # noqa

DEV = 'cuda'
TOOLBOX = {'QB': [0.34, 0.32, 0.30, 0.22], 'WV3': [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], 'WV2': [0.35] * 7 + [0.27], 'GF2': None}
MAXV = {'QB': 2047., 'GF2': 1023., 'WV3': 2047., 'WV2': 2047.}
FILES = {'QB': [('train', 'QuickBird/Training Dataset/train_qb.h5')],
         'GF2': [('train', 'Gaofen 2/Training Dataset/train_gf2.h5'), ('valid', 'Gaofen 2/Training Dataset/valid_gf2.h5')],
         'WV3': [('train', 'WorldView 3/Training Dataset/train_wv3.h5'), ('valid', 'WorldView 3/Training Dataset/valid_wv3.h5')],
         'WV2': [('test_rr', 'WorldView 2/Testing Dataset (ReducedData, H5 Format)/test_wv2_multiExm1.h5')]}
TESTS = {s: [('test_rr', f'{d}/Testing Dataset (ReducedData, H5 Format)/test_{s.lower()}_multiExm1.h5'), ('test_fr', f'{d}/Testing Dataset (FullData, H5 Format)/test_{s.lower()}_OrigScale_multiExm1.h5')]
         for s, d in (('QB', 'QuickBird'), ('GF2', 'Gaofen 2'), ('WV3', 'WorldView 3'), ('WV2', 'WorldView 2'))}
GRID = [round(x, 3) for x in np.arange(0.15, 0.501, 0.01)]


def kernel(g, size=41, ratio=4):
    c = np.arange(size) - (size - 1) / 2; y, x = np.meshgrid(c, c, indexing='ij')
    win = np.interp(np.sqrt(x * x + y * y), np.arange(size // 2 + 1), np.kaiser(size, 0.5)[size // 2:], right=0.0)
    a = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(g)))
    d = np.exp(-(x * x + y * y) / (2 * a * a)); d[d < np.finfo(d.dtype).eps * d.max()] = 0; d /= d.max()
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d))) * win)


def blur(x, gains):
    k = torch.from_numpy(np.stack([kernel(g) for g in gains])[:, None]).to(x)
    p = k.shape[-1] // 2
    r = torch.arange(-p, x.shape[-2] + p, device=x.device).clamp(0, x.shape[-2] - 1); c = torch.arange(-p, x.shape[-1] + p, device=x.device).clamp(0, x.shape[-1] - 1)
    return F.conv2d(x.index_select(-2, r).index_select(-1, c), k, groups=x.shape[1])


def rel(a, b): return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))


def audit(sensor, path, n=512):
    with h5py.File(path, 'r') as f:
        meta = {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype)) for k in f if isinstance(f[k], h5py.Dataset)}
        total = f['ms'].shape[0]; idx = np.sort(np.random.default_rng(54).choice(total, min(n, total), replace=False))
        arr = {k: torch.from_numpy(np.asarray(f[k][idx], dtype=np.float64)).to(DEV) for k in ('gt', 'ms', 'lms', 'pan') if k in f}
        stats = {k: dict(min=float(np.asarray(f[k][idx]).min()), max=float(np.asarray(f[k][idx]).max()), p99_9=float(np.percentile(np.asarray(f[k][idx]), 99.9))) for k in ('gt', 'ms', 'pan') if k in f}
    res = dict(file=str(path), meta=meta, count=total, sampled=len(idx), value_stats=stats)
    lms_hat = interp23tap((arr['ms'] / MAXV[sensor]).float()).double() * MAXV[sensor]
    res['lms_reproduction_max_abs_dn'] = float((lms_hat - arr['lms']).abs().max()); res['lms_reproduction_rel'] = rel(lms_hat, arr['lms'])
    if 'gt' not in arr: return res
    gt, ms = arr['gt'], arr['ms']; nb = gt.shape[1]
    phase = {}
    for ph in range(4):
        b = blur(gt, TOOLBOX[sensor] or [0.3] * nb)
        phase[ph] = rel(b[..., ph::4, ph::4], ms)
    best_ph = min(phase, key=phase.get); res['phase_rel_rmse_toolbox_or_0.3'] = phase; res['best_phase'] = best_ph
    if TOOLBOX[sensor]: res['toolbox_rel_rmse'] = phase[best_ph]
    uniform = {g: rel(blur(gt, [g] * nb)[..., best_ph::4, best_ph::4], ms) for g in GRID}
    res['uniform_best'] = min(uniform, key=uniform.get); res['uniform_best_rel_rmse'] = uniform[res['uniform_best']]; res['uniform_0.3_rel_rmse'] = uniform[0.3]
    per = []
    for bnd in range(nb):
        e = {g: rel(blur(gt[:, bnd:bnd + 1], [g])[..., best_ph::4, best_ph::4], ms[:, bnd:bnd + 1]) for g in GRID}
        g0 = min(e, key=e.get); per.append(dict(band=bnd, best=g0, rel_rmse=e[g0], toolbox=TOOLBOX[sensor][bnd] if TOOLBOX[sensor] else None, toolbox_rel_rmse=e.get(round(TOOLBOX[sensor][bnd], 3)) if TOOLBOX[sensor] else None))
    res['per_band'] = per
    return res


out = {}
ONLY = sys.argv[4].split(',') if len(sys.argv) > 4 else ['QB', 'GF2', 'WV3', 'WV2']
for sensor in ONLY:
    out[sensor] = {}
    SPLITS = sys.argv[5].split(',') if len(sys.argv) > 5 else None
    for split, rel_path in FILES[sensor] + [t for t in TESTS[sensor] if t[0] not in dict(FILES[sensor])]:
        if SPLITS and split not in SPLITS: continue
        p = DATA / rel_path
        if not p.exists(): out[sensor][split] = 'missing'; continue
        r = audit(sensor, p, n=512 if split in ('train', 'valid') else 10 ** 6)
        out[sensor][split] = r
        print(sensor, split, 'count', r['count'], {k: v['shape'] for k, v in r['meta'].items()}, 'max', {k: round(v['max']) for k, v in r['value_stats'].items()},
              'LMS max|d|', round(r['lms_reproduction_max_abs_dn'], 3), 'phase', r.get('best_phase'), 'toolbox', round(r.get('toolbox_rel_rmse', float('nan')), 4),
              'uniform best', r.get('uniform_best'), round(r.get('uniform_best_rel_rmse', float('nan')), 4), 'per-band', [p['best'] for p in r.get('per_band', [])], flush=True)
OUT.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
