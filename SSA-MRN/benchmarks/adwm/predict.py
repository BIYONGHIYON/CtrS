"""Save ADWM RR test predictions for MATLAB scoring: output_mulExm_<i>.mat (variable sr, DN, H x W x C) for every RR scene,
zipped as adwm_<setting>_<SENSOR>_rr.zip (AGENTS.md section 8). Uses the adapter and the prediction code of
common/benchmark_evaluate.py (load_model, predict), so the saved arrays are the ones its metrics are computed from.

python SSA-MRN/benchmarks/adwm/predict.py --setting retrain --sensor GF2 --seed 42 --data <test folder> --out <folder> --device cuda
python SSA-MRN/benchmarks/adwm/predict.py --setting author --sensor WV3 --checkpoint SSA-MRN/benchmarks/adwm/upstream/weights/BWNET_LAGConv_WV3/model.pth ...
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import benchmark_evaluate as E  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setting', required=True); p.add_argument('--sensor', choices=tuple(E.SENSORS), required=True)
    p.add_argument('--seed', type=int); p.add_argument('--checkpoint'); p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cpu')
    a = p.parse_args()
    stem, scale, _, _ = E.SENSORS[a.sensor]
    net, ck = E.load_model(HERE, a.setting, a.sensor, a.seed, a.checkpoint, a.device)
    folder = a.out / f'adwm_{a.setting}_{a.sensor}_rr'; folder.mkdir(parents=True, exist_ok=True); files = []
    for i, s in E.scenes(Path(a.data) / f'test_{stem}_multiExm1.h5', ('pan', 'lms', 'ms'), None):
        f = folder / f'output_mulExm_{i}.mat'
        scipy.io.savemat(f, {'sr': np.moveaxis(E.predict(net, s, scale, a.device), 0, -1)}); files.append(f)
    z = a.out / f'adwm_{a.setting}_{a.sensor}_rr.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files: zf.write(f, f.name)
    rec = dict(setting=a.setting, sensor=a.sensor, seed=a.seed, checkpoint=dict(path=str(ck), sha256=E.sha256(ck)), layernorm=net.layernorm,
               scenes=len(files), zip=z.name, zip_sha256=E.sha256(z), files={f.name: E.sha256(f) for f in files})
    (a.out / f'adwm_{a.setting}_{a.sensor}_rr.json').write_text(json.dumps(rec, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('setting', 'sensor', 'scenes', 'zip', 'zip_sha256', 'layernorm')}), flush=True)


if __name__ == '__main__':
    main()
