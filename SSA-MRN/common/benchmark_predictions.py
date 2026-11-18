"""Read another model's saved RR test predictions from its GitHub Release (no weights needed).

Release predictions-benchmarks-<model>-<setting>, asset <model>_<setting>_<SENSOR>_rr.zip, holding output_mulExm_<i>.mat
(variable sr, DN, H x W x C) for every RR test scene: the files the MATLAB workflow scores (AGENTS.md section 8).
The zip is downloaded once to SSA-MRN/runtime/predictions/<release>/ (ignored by Git) and checked against the SHA-256
GitHub reports for the asset. Used by common/benchmark_evaluate.py --release (result images) and viewer/export.py.

python SSA-MRN/common/benchmark_predictions.py arconv author QB     # download, check and print the array shape
"""
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
REPO = 'BIYONGHIYON/CtrS'
CACHE = ROOT / 'runtime/predictions'
SCENE = re.compile(r'(?:^|/)output_mulExm_(\d+)\.mat$')


def release_name(model, setting): return f'predictions-benchmarks-{model}-{setting}'
def asset_name(model, setting, sensor): return f'{model}_{setting}_{sensor}_rr.zip'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def _gh(*args):
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def fetch(model, setting, sensor, cache=CACHE):
    """Local path of the checked zip and its release/asset/SHA-256 record."""
    tag, name = release_name(model, setting), asset_name(model, setting, sensor)
    assets = {a['name']: a for a in json.loads(_gh('api', f'repos/{REPO}/releases/tags/{tag}'))['assets']}
    if name not in assets: raise FileNotFoundError(f'{tag} has no asset {name} (has: {", ".join(sorted(assets))})')
    want = assets[name]['digest'].removeprefix('sha256:')
    path = Path(cache) / tag / name
    if not path.is_file() or _sha(path) != want:
        path.parent.mkdir(parents=True, exist_ok=True)
        _gh('release', 'download', tag, '-R', REPO, '-p', name, '-D', str(path.parent), '--clobber')
    got = _sha(path)
    if got != want: raise ValueError(f'{name}: SHA-256 {got} != GitHub {want}')
    return path, dict(release=tag, asset=name, asset_sha256=got)


def _read(data, key):
    try:
        x = scipy.io.loadmat(io.BytesIO(data))[key]
    except NotImplementedError:                      # MATLAB v7.3 (HDF5): stored column-major, so H x W x C reads as C x W x H
        import h5py
        with h5py.File(io.BytesIO(data), 'r') as f: return np.asarray(f[key], dtype=np.float64).transpose(0, 2, 1)
    return np.moveaxis(np.asarray(x, dtype=np.float64), -1, 0)


def load(model, setting, sensor, key='sr', cache=CACHE):
    """Predictions (N, C, H, W) in DN, float64, scene order 0..N-1, and the release record."""
    path, info = fetch(model, setting, sensor, cache)
    with zipfile.ZipFile(path) as z:
        files = {int(m.group(1)): n for n in z.namelist() if (m := SCENE.search(n))}
        if sorted(files) != list(range(len(files))): raise ValueError(f'{path.name}: scenes {sorted(files)} are not 0..N-1')
        pred = np.stack([_read(z.read(files[i]), key) for i in range(len(files))])
    if not np.isfinite(pred).all(): raise ValueError(f'{path.name}: nonfinite values')
    return pred, info


if __name__ == '__main__':
    p, info = load(*sys.argv[1:4])
    print(p.shape, f'{p.min():.1f}..{p.max():.1f} DN', info)
