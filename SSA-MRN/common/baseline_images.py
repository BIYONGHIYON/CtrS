"""Root README / docs/c2ms_sp_result_images.md material for the baseline comparison: original SSA-MRN, C2MS-SP, C2MS-SP-Lite.

Writes docs/images/c2ms_sp/:
  baseline_metrics.json  RR test metrics (PSNR, SAM, ERGAS, SCC, Q2n) of the three models on QB, GF2, WV3 (each trained on its own sensor) and
                         the WV2 transfer (WV3 weights), seeds 42-44, per seed and 3-seed mean; on WV2 also C2MS-SP / C2MS-SP-Lite without the
                         band calibration, and the transfer detail measures of experiment 54 (sharpness = mean |4-neighbour Laplacian| of the
                         output / of the GT, detail correlation of (output - LMS) with (GT - LMS), per-band mean bias / GT mean)
  <dataset>/full_scene_XX.png, compare_scene_XX.png, result_images.json  fixed six scenes, seed 42 (common/result_images.render_multi)
Models come from common/baseline_models (C2MS-SP and -Lite include the K10 projection; on WV2 also the band calibration of experiment 82;
the original SSA-MRN runs as published). Usage: python SSA-MRN/common/baseline_images.py [--device cpu] [--skip-metrics] [--skip-images]
"""
import argparse, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import baseline_models as B  # noqa: E402
from common import result_images  # noqa: E402
from common.metrics import rr_metrics  # noqa: E402

OUT = ROOT / 'docs/images/c2ms_sp'
DATASETS = {'QB': ('test_qb_multiExm1.h5', 2047., [2, 1, 0]), 'GF2': ('test_gf2_multiExm1.h5', 1023., [2, 1, 0]),
            'WV3': ('test_wv3_multiExm1.h5', 2047., [4, 2, 1]), 'WV2': ('test_wv2_multiExm1.h5', 2047., [4, 2, 1])}
SEEDS = (42, 43, 44); NAMES = ('original', 'c2ms_sp', 'c2ms_sp_lite'); KEYS = ('PSNR', 'SAM', 'ERGAS', 'SCC', 'Q2n')


def lap(x): return np.abs(4 * x[:, 1:-1, 1:-1] - x[:, :-2, 1:-1] - x[:, 2:, 1:-1] - x[:, 1:-1, :-2] - x[:, 1:-1, 2:]).mean()


def run(net, rr, scale):
    ys = []
    with torch.no_grad():
        for a in rr:
            x = [torch.from_numpy((a[k] / scale).astype('float32'))[None] for k in ('pan', 'lms', 'ms')]
            ys.append(net(*x)[0].double().cpu().numpy() * scale)
    return ys


def metrics(device):
    res = {}
    for ds, (f, scale, _) in DATASETS.items():
        with h5py.File(ROOT / 'data/test' / f) as h: rr = [{k: np.asarray(h[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
        variants = [(n, True) for n in NAMES] + ([('c2ms_sp', False), ('c2ms_sp_lite', False)] if ds == 'WV2' else [])
        for name, cal in variants:
            key = name + ('' if cal or ds != 'WV2' or name == 'original' else '_no_calibration'); per = {}
            for seed in SEEDS:
                t = time.time(); ys = run(B.build(name, ds, seed, calibrate=cal, device=device), rr, scale)
                m = [rr_metrics(a['gt'], y) for a, y in zip(rr, ys)]; r = {k: float(np.mean([x[k] for x in m])) for k in KEYS}
                if ds == 'WV2':
                    gm = np.mean([a['gt'].mean((1, 2)) for a in rr], 0)
                    r.update(sharpness=float(np.mean([lap(y) / lap(a['gt']) for a, y in zip(rr, ys)])),
                             detail_corr=float(np.mean([np.corrcoef((y - a['lms']).ravel(), (a['gt'] - a['lms']).ravel())[0, 1] for a, y in zip(rr, ys)])),
                             band_bias=((np.mean([y.mean((1, 2)) for y in ys], 0) - gm) / gm).tolist())
                per[seed] = r; print(ds, key, seed, {k: round(v, 4) for k, v in r.items() if not isinstance(v, list)}, f'{time.time() - t:.0f}s', flush=True)
            mean = {k: float(np.mean([per[s][k] for s in SEEDS])) for k in per[SEEDS[0]] if k != 'band_bias'}
            if ds == 'WV2':
                bias = np.mean([per[s]['band_bias'] for s in SEEDS], 0); mean['band_bias'] = bias.tolist(); mean['max_abs_band_bias'] = float(np.abs(bias).max())
                mean['max_abs_band_bias_per_seed'] = [float(np.abs(per[s]['band_bias']).max()) for s in SEEDS]
            res.setdefault(ds, {})[key] = dict(per_seed=per, mean=mean)
    return res


def images(device):
    labels = [B.LABELS[n] for n in NAMES]
    for ds, (f, scale, bands) in DATASETS.items():
        nets = {n: B.build(n, ds, 42, calibrate=True, device=device) for n in NAMES}
        lab = labels if ds != 'WV2' else [labels[0], labels[1] + ' (보정)', labels[2] + ' (보정)']
        result_images.render_multi(nets, ROOT / 'data/test', OUT / ds, lab, seed=42, device=device, rr_file=f, scale=scale, bands=bands,
                                   land_cover=result_images.LAND_COVER if ds == 'QB' else {i: f'{ds} 장면 {i}' for i in result_images.SCENES}, chooser=1)
        print('images', ds, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); p.add_argument('--skip-metrics', action='store_true'); p.add_argument('--skip-images', action='store_true')
    a = p.parse_args(); torch.set_num_threads(8); OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_images: images(a.device)
    if not a.skip_metrics:
        res = metrics(a.device)
        (OUT / 'baseline_metrics.json').write_text(json.dumps(dict(description=__doc__, seeds=SEEDS, results=res), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
