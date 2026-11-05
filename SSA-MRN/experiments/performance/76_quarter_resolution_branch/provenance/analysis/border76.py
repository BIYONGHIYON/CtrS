"""76 follow-up: split the error change of each W4 arm by frequency band and by distance to the image border (16 px strip vs interior).
Reads the prediction cache written by analyze76.py; QB RR test, seed 42."""
import json, sys
from pathlib import Path
import numpy as np, h5py
SCR = Path(__file__).resolve().parent; CACHE = SCR / 'cache76'; OUT = SCR / 'analysis76'
gt = np.asarray(h5py.File('/Users/biyonghiyon_e/Desktop/CtrS/SSA-MRN/data/test/test_qb_multiExm1.h5', 'r')['gt'], dtype=np.float64)
r = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing='ij'))
bands = {'>16px': r < 1 / 16, '8-16px': (r >= 1 / 16) & (r < 1 / 8), '4-8px': (r >= 1 / 8) & (r < 1 / 4), '<4px': r >= 1 / 4}
yy, xx = np.mgrid[:256, :256]; d = np.minimum(np.minimum(yy, 255 - yy), np.minimum(xx, 255 - xx))
regions = {'border_16px': d < 16, 'interior': d >= 16}
Y = {n: np.load(CACHE / f'QB_{n}_s42.npy').astype(np.float64) for n in ('C2MS', 'B', 'W64D8', 'A')}
def band_err(y, m): return np.fft.ifft2(np.fft.fft2(y - gt, axes=(-2, -1)) * m, axes=(-2, -1)).real ** 2
res = dict(pixel_share={k: float(v.mean()) for k, v in regions.items()}, pairs={})
for a, b in (('C2MS', 'B'), ('W64D8', 'A')):
    row = {}
    for bn, m in bands.items():
        ea, eb = band_err(Y[a], m).mean((0, 1)), band_err(Y[b], m).mean((0, 1))
        row[bn] = {rn: dict(mse_reduction_pct=float(100 * (1 - eb[rm].mean() / ea[rm].mean())), se_drop=float(ea[rm].sum() - eb[rm].sum())) for rn, rm in regions.items()}
    ea, eb = ((Y[a] - gt) ** 2).mean((0, 1)), ((Y[b] - gt) ** 2).mean((0, 1))
    row['all'] = {rn: dict(mse_reduction_pct=float(100 * (1 - eb[rm].mean() / ea[rm].mean())), share_of_total_se_drop=float((ea[rm].sum() - eb[rm].sum()) / (ea.sum() - eb.sum()))) for rn, rm in regions.items()}
    res['pairs'][f'{a}->{b}'] = row
json.dump(res, open(OUT / 'border76.json', 'w'), indent=1); print(json.dumps(res['pairs']['C2MS->B']['all']))
