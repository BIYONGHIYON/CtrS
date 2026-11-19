"""Whole-image inference (adapter, Windows-capable) vs the official tiled inference (PatchMergeModule, Linux .so).

Needs Linux x86_64 with Python 3.7-3.10 or 3.12 (the .so files in upstream/models/patch_merge_module/). Runs the WV3 author
weights on every RR and FR test scene both ways and records how far the outputs are apart. It does not compute quality
metrics: the decision rule is fixed before running (README.md, 방법) and does not look at which way scores higher.

python SSA-MRN/benchmarks/lformer/compare_patch_merge.py --rr data/test_wv3_multiExm1.h5 --fr data/test_wv3_OrigScale_multiExm1.h5 --device cuda
-> results/patch_merge_check.json
"""
import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCALE = 2047.0
sys.path.insert(0, str(HERE.parents[1]))
from common.metrics import rr_metrics  # noqa: E402


def adapter():
    spec = importlib.util.spec_from_file_location('lformer_adapter', HERE / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(net, h, i, device):
    x = [torch.from_numpy(np.asarray(h[k][i], dtype=np.float32)[None] / SCALE).to(device) for k in ('pan', 'lms', 'ms')]
    with torch.no_grad(): return net(*x)[0].double().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rr', required=True, type=Path); p.add_argument('--fr', required=True, type=Path)
    p.add_argument('--device', default='cuda'); p.add_argument('--limit', type=int)
    a = p.parse_args()
    mod = adapter()
    nets = {mode: mod.load(mod.build(8, 'WV3', 'author', None, patch_merge=mode == 'tiled'), mod.AUTHOR_WEIGHTS).to(a.device).eval()
            for mode in ('whole', 'tiled')}
    report = dict(weights=str(mod.AUTHOR_WEIGHTS.relative_to(HERE)), python=platform.python_version(), torch=str(torch.__version__),
                  device=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else a.device,
                  note='outputs in [0, 1] (DN / 2047); max_abs_dn = max |whole - tiled| x 2047', tests={})
    for name, path in (('RR', a.rr), ('FR', a.fr)):
        rows = []
        with h5py.File(path, 'r') as h:
            n = h['pan'].shape[0] if a.limit is None else min(a.limit, h['pan'].shape[0])
            for i in range(n):
                w, t = run(nets['whole'], h, i, a.device), run(nets['tiled'], h, i, a.device)
                d = np.abs(w - t)
                row = dict(scene=i, max_abs=float(d.max()), mean_abs=float(d.mean()), max_abs_dn=float(d.max() * SCALE),
                           changed_pixels=float((d > 0.5 / SCALE).mean()))   # > half a DN
                if 'gt' in h:
                    gt = np.asarray(h['gt'][i], dtype=np.float64)
                    row.update(whole=rr_metrics(gt, w * SCALE), tiled=rr_metrics(gt, t * SCALE))
                rows.append(row)
                print(name, rows[-1], flush=True)
        report['tests'][name] = dict(file=path.name, scenes=rows, max_abs_dn=max(r['max_abs_dn'] for r in rows),
                                     mean_abs_dn=float(np.mean([r['mean_abs'] for r in rows]) * SCALE))
        if rows and 'whole' in rows[0]:
            report['tests'][name]['mean_metrics'] = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in rows[0][m]}
                                                    for m in ('whole', 'tiled')}
    report['checked'] = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    out = HERE / 'results/patch_merge_check.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f"RR max |diff| {report['tests']['RR']['max_abs_dn']:.3f} DN, FR {report['tests']['FR']['max_abs_dn']:.3f} DN -> {out}")


if __name__ == '__main__':
    sys.exit(main())
