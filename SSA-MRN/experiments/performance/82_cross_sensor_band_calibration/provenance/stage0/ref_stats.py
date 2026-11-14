"""Channel means of the training-sensor data (DN): QB from the full training set; GF2/WV3 from the RR test inputs (training sets are not local)."""
import json, sys
from pathlib import Path
import h5py, numpy as np
ROOT = Path(__file__).resolve().parents[5]; out = {}
with h5py.File(ROOT / 'data/train/train_qb.h5') as f:
    n = len(f['pan']); sp = 0.; sl = 0.; rp = []; rb = []
    for s in range(0, n, 2048):
        pan = np.asarray(f['pan'][s:s + 2048], dtype=np.float64); lms = np.asarray(f['lms'][s:s + 2048], dtype=np.float64)
        sp += pan.sum(); sl = sl + lms.sum((0, 2, 3)); rp.append(pan.mean((1, 2, 3))); rb.append(lms.mean((2, 3)))
    px = n * 64 * 64; rp = np.concatenate(rp); rb = np.concatenate(rb)
    out['QB_train'] = dict(n=n, pan=sp / px, bands=(sl / px).tolist(), patch_ratio_mean=(rb / rp[:, None]).mean(0).tolist(), ratio_of_means=((sl / px) / (sp / px)).tolist(), keys=list(f.keys()), max=float(np.asarray(f['pan'][:256]).max()))
for s, tag in (('QB', 'qb'), ('GF2', 'gf2'), ('WV3', 'wv3'), ('WV2', 'wv2')):
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        pan = np.asarray(f['pan'][:], dtype=np.float64); lms = np.asarray(f['lms'][:], dtype=np.float64)
        out[f'{s}_test'] = dict(n=len(pan), pan=float(pan.mean()), bands=lms.mean((0, 2, 3)).tolist(), ratio_of_means=(lms.mean((0, 2, 3)) / pan.mean()).tolist(),
                                scene_ratio_mean=(lms.mean((2, 3)) / pan.mean((1, 2, 3))[:, None]).mean(0).tolist(), scene_ratio_sd=(lms.mean((2, 3)) / pan.mean((1, 2, 3))[:, None]).std(0).tolist(), scene_pan_sd=float(pan.mean((1, 2, 3)).std()))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(k, {a: (np.round(np.asarray(b, dtype=float), 3).tolist() if a != 'keys' else b) for a, b in v.items()})
