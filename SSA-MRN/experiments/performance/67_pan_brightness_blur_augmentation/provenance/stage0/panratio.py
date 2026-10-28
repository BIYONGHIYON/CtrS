"""PAN / MS intensity ratio of the training sets (per patch: mean(PAN) / mean(LMS)); no test data. Reads in chunks."""
import json, sys
from pathlib import Path
import h5py, numpy as np
root = Path(sys.argv[1]); out = {}
for folder, tag in (('QuickBird', 'qb'), ('Gaofen 2', 'gf2'), ('WorldView 3', 'wv3')):
    for split, name in (('train', f'train_{tag}.h5'), ('valid', f'valid_{tag}.h5')):
        p = root / folder / 'Training Dataset' / name
        if not p.exists(): continue
        with h5py.File(p, 'r') as f:
            n = len(f['pan']); r = []
            for s in range(0, n, 2048):
                pan = np.asarray(f['pan'][s:s + 2048], dtype=np.float64); lms = np.asarray(f['lms'][s:s + 2048], dtype=np.float64)
                r.append(pan.mean((1, 2, 3)) / np.maximum(lms.mean((1, 2, 3)), 1e-6))
            r = np.concatenate(r)
        out[f'{tag}_{split}'] = dict(n=int(n), mean=float(r.mean()), p05=float(np.percentile(r, 5)), p50=float(np.median(r)), p95=float(np.percentile(r, 95)))
        print(tag, split, out[f'{tag}_{split}'], flush=True)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
