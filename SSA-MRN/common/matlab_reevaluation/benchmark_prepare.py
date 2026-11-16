"""Prepare another model's saved RR predictions for actual-MATLAB scoring with score_suite.m (no inference here).

Some benchmark models only run on CUDA (e.g. CANNet's custom operators), so their predictions are made elsewhere (Kaggle GPU) and
scored here. Input: a folder of per-scene .mat files in DN, H x W x C (the official CANConv test script writes
output_mulExm_<i>.mat with key 'sr'). The exact original RR test file is downloaded and SHA-256 checked as in suite.py
(data_files), each scene is written as fused/lms/ms/pan/reference (H x W x C, DN) for score_suite.m, and the same arrays are scored
with the CtrS Python metrics (common/metrics) so that MATLAB and Python can be compared on identical inputs.

python SSA-MRN/common/matlab_reevaluation/benchmark_prepare.py --model cannet --setting author --sensor QB \
    --predictions pred/QB --data /tmp/pancollection --out evaluation
"""
import argparse
import platform
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from suite import DLPAN_REV, data_files, download, sha, write  # noqa: E402
sys.path.insert(0, str(HERE.parents[1]))
from common.metrics import rr_metrics  # noqa: E402

BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8, 'WV2': 8}


def scene_files(folder):
    found = {}
    for p in Path(folder).rglob('*.mat'):
        m = re.search(r'(\d+)\.mat$', p.name)
        if m: found[int(m.group(1))] = p
    if sorted(found) != list(range(20)): raise ValueError(f'expected 20 scene files 0-19, got {sorted(found)}')
    return [found[i] for i in range(20)]


def load_fused(path, key):
    x = loadmat(path)
    keys = [k for k in x if not k.startswith('__')]
    k = key if key in x else (keys[0] if len(keys) == 1 else None)
    if k is None: raise KeyError(f'{path}: cannot find prediction key {key} in {keys}')
    return np.asarray(x[k], dtype=np.float64)  # H x W x C, DN


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True); p.add_argument('--setting', default='author')
    p.add_argument('--sensor', choices=tuple(BANDS), required=True)
    p.add_argument('--predictions', type=Path, required=True, help='folder with 20 per-scene .mat files (DN, H x W x C)')
    p.add_argument('--key', default='sr', help='variable name of the prediction in each .mat (CANConv: sr)')
    p.add_argument('--data', type=Path, required=True, help='download cache for the original test files')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    spec = data_files(a.sensor)['RR']
    rr_file = download(spec, a.data / spec['name'])
    peak, bits = (1023, 10) if a.sensor == 'GF2' else (2047, 11)
    out = a.out.resolve(); (out / 'predictions').mkdir(parents=True, exist_ok=True)
    files, records = [], []
    with h5py.File(rr_file, 'r') as h:
        if len(h['gt']) != 20: raise ValueError('full RR20 required')
        for i, src in enumerate(scene_files(a.predictions)):
            s = {k: np.asarray(h[k][i], dtype=np.float64) for k in ('gt', 'pan', 'lms', 'ms')}
            fused = load_fused(src, a.key)
            c, hh, ww = s['gt'].shape
            if c != BANDS[a.sensor] or fused.shape != (hh, ww, c): raise ValueError(f'scene {i}: prediction {fused.shape} vs reference {(hh, ww, c)}')
            if not np.isfinite(fused).all(): raise FloatingPointError(f'scene {i}: nonfinite prediction')
            records.append(dict(protocol='RR', scene=i, python=rr_metrics(s['gt'], fused.transpose(2, 0, 1)), source=src.name, source_sha256=sha(src)))
            path = out / 'predictions' / f'RR_{i:02}.mat'
            savemat(path, dict(fused=fused, lms=s['lms'].transpose(1, 2, 0), ms=s['ms'].transpose(1, 2, 0),
                               pan=s['pan'].transpose(1, 2, 0), reference=s['gt'].transpose(1, 2, 0)), do_compression=False)
            files.append(dict(protocol='RR', scene=i, file='predictions/' + path.name, sha256=sha(path)))
    case = dict(id=f'{a.model}_{a.setting}_{a.sensor}', model=a.model, setting=a.setting, sensor=a.sensor, experiment='benchmark')
    write(out / 'prediction_manifest.json', dict(case=case, files=files, scenes=20, peak=peak, bits=bits, ratio=4, block=32,
          dlpan_commit=DLPAN_REV, data=dict(RR=dict(spec, sha256_checked=sha(rr_file))),
          inference='made elsewhere (see the model report); this step only reads the saved predictions',
          environment=dict(python=platform.python_version(), numpy=np.__version__)))
    write(out / 'python_metrics.json', records)
    mean = {k: float(np.mean([r['python'][k] for r in records])) for k in records[0]['python']}
    print(case['id'], 'Python', {k: round(v, 4) for k, v in mean.items()}, flush=True)


if __name__ == '__main__':
    main()
