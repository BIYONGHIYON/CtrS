"""77 follow-up: split the error change of each enlarged-backbone arm by frequency band and by distance to the image border
(16 px strip vs interior), as border76.py. Reads the prediction cache written by analyze77.py; QB RR test, seed 42."""
import json
from pathlib import Path
import numpy as np, h5py
SCR = Path(__file__).resolve().parent; CACHE = SCR / 'cache77'; OUT = SCR / 'analysis77'
gt = np.asarray(h5py.File('/Users/biyonghiyon_e/Desktop/CtrS/SSA-MRN/data/test/test_qb_multiExm1.h5', 'r')['gt'], dtype=np.float64)
r = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing='ij'))
bands = {'>16px': r < 1 / 16, '8-16px': (r >= 1 / 16) & (r < 1 / 8), '4-8px': (r >= 1 / 8) & (r < 1 / 4), '<4px': r >= 1 / 4}
yy, xx = np.mgrid[:256, :256]; d = np.minimum(np.minimum(yy, 255 - yy), np.minimum(xx, 255 - xx))
regions = {'border_16px': d < 16, 'interior': d >= 16}
Y = {n: np.load(CACHE / f'QB_{n}_s42.npy').astype(np.float64) for n in ('C2MS', 'E1', 'E2', 'W64D8', 'E3')}
def band_err(y, m): return np.fft.ifft2(np.fft.fft2(y - gt, axes=(-2, -1)) * m, axes=(-2, -1)).real ** 2
res = dict(pixel_share={k: float(v.mean()) for k, v in regions.items()}, pairs={})
for a, b in (('C2MS', 'E1'), ('C2MS', 'E2'), ('W64D8', 'E3'), ('C2MS', 'E3')):
    row = {}
    for bn, m in bands.items():
        ea, eb = band_err(Y[a], m).mean((0, 1)), band_err(Y[b], m).mean((0, 1))
        row[bn] = {rn: dict(mse_reduction_pct=float(100 * (1 - eb[rm].mean() / ea[rm].mean())), se_drop=float(ea[rm].sum() - eb[rm].sum())) for rn, rm in regions.items()}
    ea, eb = ((Y[a] - gt) ** 2).mean((0, 1)), ((Y[b] - gt) ** 2).mean((0, 1))
    row['all'] = {rn: dict(mse_reduction_pct=float(100 * (1 - eb[rm].mean() / ea[rm].mean())), share_of_total_se_drop=float((ea[rm].sum() - eb[rm].sum()) / (ea.sum() - eb.sum()))) for rn, rm in regions.items()}
    res['pairs'][f'{a}->{b}'] = row
json.dump(res, open(OUT / 'border_interior.json', 'w'), indent=1)
for k, v in res['pairs'].items(): print(k, json.dumps({b: {rn: round(x['mse_reduction_pct'], 2) for rn, x in v[b].items()} for b in v}))
