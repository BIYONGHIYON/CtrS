"""Evaluate another pansharpening model (SSA-MRN/benchmarks/<model>/) under the same conditions as the baseline SMOC.

Adapter (benchmarks/<model>/model.py) must define
    build(bands, sensor, setting, seed) -> torch.nn.Module
        forward(pan [B,1,H,W], lms [B,C,H,W], ms [B,C,H/4,W/4]) -> [B,C,H,W]; inputs and output are DN / scale
        (QB, WV3, WV2: 2047; GF2: 1023). Any reordering, padding or rescaling the original code needs belongs in the adapter.
and may define
    load(net, path)  load a checkpoint into net (default: torch.load(path)['model'] or the whole file, strict load)

Checkpoints: benchmarks/<model>/checkpoints/<setting>_<train sensor>_s<seed>/best.pt (author weights without a seed:
<setting>_<train sensor>/best.pt) or --checkpoint. Settings: author (published weights, no training), retrain (official code and
paper settings trained on our split; the published number of epochs, or the 150 epochs of SMOC when none is published);
see benchmarks/README.md.

Writes benchmarks/<model>/results/<setting>_<sensor>_s<seed>/metrics.json: per-scene RR metrics (PSNR, SAM, ERGAS, SCC, Q2n;
common/metrics, the code the baseline numbers come from) on all RR test scenes, per-scene FR QNR / D_lambda / D_s (reference),
checkpoint and data SHA-256, parameter count. WV2 is evaluated with the WV3 weights (transfer, reference only).
--summary pools the runs into results/summary_smoc.json: per setting and sensor the seed mean and the paired difference to SMOC
(same seeds, docs/images/smoc/baseline_metrics.json). No judgement is attached; the numbers are reported as they are.
--baseline c2ms_sp reproduces archived comparisons (summary.json and images/); the default SMOC writes summary_smoc.json and images_smoc/.
SMOC is the factorized model (90 CK) since 2026-11-24; comparisons archived before that date are against the full-size SMOC (90 CF), which
--baseline smoc_full reproduces (docs/images/smoc_full/baseline_metrics.json).
--figures renders the fixed six scenes in both formats (common/result_images.render_multi): SMOC, then the model.
--figures --release takes the model's saved predictions from Release predictions-benchmarks-<model>-<setting> instead of running
its weights (common/benchmark_predictions.py), so images can be made for author weights that are not redistributed.

python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --device cuda
python SSA-MRN/common/benchmark_evaluate.py --model adwm --summary
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --figures --device cuda
python SSA-MRN/common/benchmark_evaluate.py --model arconv --setting author --sensor GF2 --figures --release
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.metrics import fr_metrics, rr_metrics  # noqa: E402

SENSORS = {'QB': ('qb', 2047., 4, [2, 1, 0]), 'GF2': ('gf2', 1023., 4, [2, 1, 0]),
           'WV3': ('wv3', 2047., 8, [4, 2, 1]), 'WV2': ('wv2', 2047., 8, [4, 2, 1])}
TRAINING_SENSOR = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}  # WV2 has no training data: WV3 weights (transfer)
RR_KEYS = ('PSNR', 'SAM', 'ERGAS', 'SCC', 'Q2n'); FR_KEYS = ('QNR', 'D_lambda', 'D_s')
BASELINES = {
    'smoc': ROOT / 'docs/images/smoc/baseline_metrics.json',
    'smoc_full': ROOT / 'docs/images/smoc_full/baseline_metrics.json',
    'c2ms_sp': ROOT / 'docs/images/c2ms_sp/baseline_metrics.json',
}
BASELINE_LABELS = {'smoc': 'SMOC', 'smoc_full': 'SMOC', 'c2ms_sp': 'C2MS-SP'}
BASELINE_KEY = {'smoc_full': 'smoc'}  # the archived full-size metrics file stores SMOC under 'smoc'
KST = timezone(timedelta(hours=9))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda: s.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def adapter(folder):
    spec = importlib.util.spec_from_file_location(f'benchmark_{folder.name}', folder / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if not hasattr(module, 'build'): raise SystemExit(f'{folder}/model.py has no build(bands, sensor, setting, seed)')
    return module


def run_name(setting, sensor, seed): return f'{setting}_{sensor}' + ('' if seed is None else f'_s{seed}')


def load_model(folder, setting, sensor, seed, checkpoint=None, device='cpu'):
    train = TRAINING_SENSOR[sensor]; bands = SENSORS[train][2]; module = adapter(folder)
    net = module.build(bands, sensor, setting, seed)
    path = Path(checkpoint) if checkpoint else folder / 'checkpoints' / run_name(setting, train, seed) / 'best.pt'
    if not path.is_file(): raise SystemExit(f'checkpoint not found: {path}')
    if hasattr(module, 'load'): module.load(net, path)
    else:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck, strict=True)
    return net.to(device).eval(), path


def predict(net, scene, scale, device):
    x = [torch.from_numpy((scene[k] / scale).astype('float32'))[None].to(device) for k in ('pan', 'lms', 'ms')]
    with torch.no_grad(): y = net(*x)
    out = y[0].double().cpu().numpy() * scale
    if out.shape != scene['lms'].shape: raise ValueError(f'output shape {out.shape} != {scene["lms"].shape}')
    if not np.isfinite(out).all(): raise FloatingPointError('nonfinite prediction')
    return out


def scenes(path, keys, limit):
    with h5py.File(path, 'r') as h:
        n = h[keys[0]].shape[0] if limit is None else min(limit, h[keys[0]].shape[0])
        for i in range(n): yield i, {k: np.asarray(h[k][i], dtype=np.float64) for k in keys}


def mean(rows, keys): return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate(a, folder):
    stem, scale, _, _ = SENSORS[a.sensor]; data = Path(a.data)
    rr_file, fr_file = data / f'test_{stem}_multiExm1.h5', data / f'test_{stem}_OrigScale_multiExm1.h5'
    net, ck = load_model(folder, a.setting, a.sensor, a.seed, a.checkpoint, a.device); t0 = time.time()
    rr = [dict(scene=i, **rr_metrics(s['gt'], predict(net, s, scale, a.device))) for i, s in scenes(rr_file, ('pan', 'lms', 'ms', 'gt'), a.limit)]
    print('RR', a.sensor, {k: round(v, 4) for k, v in mean(rr, RR_KEYS).items()}, f'{time.time() - t0:.0f}s', flush=True)
    fr = []
    if not a.skip_fr:
        for i, s in scenes(fr_file, ('pan', 'lms', 'ms'), a.limit):
            fr.append(dict(scene=i, **fr_metrics(predict(net, s, scale, a.device), s['lms'], s['ms'], s['pan'])))
        print('FR', a.sensor, {k: round(v, 4) for k, v in mean(fr, FR_KEYS).items()}, f'{time.time() - t0:.0f}s', flush=True)
    raw = torch.load(ck, map_location='cpu', weights_only=False)
    out = dict(model=folder.name, setting=a.setting, sensor=a.sensor, training_sensor=TRAINING_SENSOR[a.sensor], seed=a.seed,
               transfer=a.sensor != TRAINING_SENSOR[a.sensor], complete=a.limit is None and not a.skip_fr,
               checkpoint=dict(path=ck.relative_to(ROOT).as_posix() if ck.is_relative_to(ROOT) else str(ck), bytes=ck.stat().st_size,
                               sha256=sha256(ck), epoch=raw.get('epoch') if isinstance(raw, dict) else None),
               data={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in ([rr_file] + ([] if a.skip_fr else [fr_file]))},
               scale=scale, parameters=sum(p.numel() for p in net.parameters()), device=a.device, evaluated=datetime.now(KST).isoformat(timespec='seconds'),
               metric_code='common/metrics.py (rr_metrics, fr_metrics)', inputs='DN / scale, float32; output x scale',
               rr=dict(scenes=len(rr), mean=mean(rr, RR_KEYS), per_scene=rr),
               fr=dict(scenes=len(fr), mean=mean(fr, FR_KEYS) if fr else None, per_scene=fr, note='reference only (criteria v2.4)'))
    path = folder / 'results' / run_name(a.setting, a.sensor, a.seed) / 'metrics.json'
    if a.limit is not None: path = path.with_name('metrics_smoke.json')
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('wrote', path)


def summary(folder, baseline='smoc'):
    baseline_path = BASELINES[baseline]; label = BASELINE_LABELS[baseline]; comparison = f'vs_{BASELINE_KEY.get(baseline, baseline)}'
    base = json.loads(baseline_path.read_text(encoding='utf-8'))['results']; groups = {}
    for p in sorted((folder / 'results').glob('*/metrics.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r['complete']: groups.setdefault((r['setting'], r['sensor']), []).append(r)
    res = {}
    for (setting, sensor), runs in sorted(groups.items()):
        runs.sort(key=lambda r: (r['seed'] is None, r['seed']))
        per = {str(r['seed']): dict(rr=r['rr']['mean'], fr=r['fr']['mean']) for r in runs}
        entry = dict(seeds=[r['seed'] for r in runs], transfer=runs[0]['transfer'], parameters=runs[0]['parameters'], per_seed=per,
                     mean=dict(rr=mean([r['rr']['mean'] for r in runs], RR_KEYS), fr=mean([r['fr']['mean'] for r in runs], FR_KEYS)))
        ref = base[sensor][BASELINE_KEY.get(baseline, baseline)]['per_seed']
        paired = [r for r in runs if r['seed'] is not None and str(r['seed']) in ref]
        if paired:
            d = [{k: r['rr']['mean'][k] - ref[str(r['seed'])][k] for k in RR_KEYS} for r in paired]
            diff = mean(d, RR_KEYS)
            entry[comparison] = dict(paired_seeds=[r['seed'] for r in paired], difference=diff, per_seed=dict(zip([str(r['seed']) for r in paired], d)),
                                       note='WV2 전이, 참고' if runs[0]['transfer'] else None)
        elif runs:
            m = entry['mean']['rr']; c = base[sensor][BASELINE_KEY.get(baseline, baseline)]['mean']
            entry[comparison] = dict(paired_seeds=[], difference={k: m[k] - c[k] for k in RR_KEYS},
                                     note=f'같은 시드 짝 없음: {label} 3시드 평균과의 차이')
        res[f'{setting}_{sensor}'] = entry
        v = entry.get(comparison, {})
        print(f"{setting} {sensor} seeds {entry['seeds']}: PSNR {entry['mean']['rr']['PSNR']:.3f} SAM {entry['mean']['rr']['SAM']:.3f} "
              f"ERGAS {entry['mean']['rr']['ERGAS']:.3f} | - {label}: PSNR {v.get('difference', {}).get('PSNR', float('nan')):+.3f} dB "
              f"SAM {v.get('difference', {}).get('SAM', float('nan')):+.3f} ERGAS {v.get('difference', {}).get('ERGAS', float('nan')):+.3f}")
    out = dict(model=folder.name, description=f'seed means and the paired difference to {label} (same seeds); common/benchmark_evaluate.py --summary',
               baseline=baseline_path.relative_to(ROOT).as_posix(), created=datetime.now(KST).isoformat(timespec='seconds'), results=res)
    filename = 'summary_smoc.json' if baseline in ('smoc', 'smoc_full') else 'summary.json'
    (folder / 'results' / filename).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def figures(a, folder):
    from common import baseline_models as B
    from common import result_images
    stem, scale, _, bands = SENSORS[a.sensor]; seed = 42 if a.seed is None else a.seed
    if a.release:
        from common import benchmark_predictions
        net, info = benchmark_predictions.load(folder.name, a.setting, a.sensor)
        print(f"{info['release']}/{info['asset']} sha256 {info['asset_sha256']}", flush=True)
    else:
        net, _ = load_model(folder, a.setting, a.sensor, a.seed, a.checkpoint, a.device)
    name = json.loads((folder / 'config.json').read_text(encoding='utf-8')).get('display_name', folder.name) if (folder / 'config.json').exists() else folder.name
    nets = {a.baseline: B.build(a.baseline, a.sensor, seed, device=a.device), folder.name: net}
    image_dir = 'images_smoc' if a.baseline in ('smoc', 'smoc_full') else 'images'
    result_images.render_multi(nets, Path(a.data), folder / 'results' / image_dir / run_name(a.setting, a.sensor, a.seed), [BASELINE_LABELS[a.baseline], name], seed=seed,
                               device=a.device, rr_file=f'test_{stem}_multiExm1.h5', scale=scale, bands=bands,
                               land_cover=result_images.LAND_COVER if a.sensor == 'QB' else {i: f'{a.sensor} 장면 {i}' for i in result_images.SCENES}, chooser=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', help='folder name under SSA-MRN/benchmarks/')
    p.add_argument('--folder', type=Path, help='model folder (default SSA-MRN/benchmarks/<model>)')
    p.add_argument('--setting', default='retrain', help='author / retrain or a setting defined in config.json')
    p.add_argument('--sensor', choices=tuple(SENSORS), default='QB', help='test sensor; WV2 uses the WV3 weights')
    p.add_argument('--seed', type=int, help='training seed (omit for author weights)')
    p.add_argument('--checkpoint', help='checkpoint path instead of checkpoints/<setting>_<sensor>_s<seed>/best.pt')
    p.add_argument('--data', default=str(ROOT / 'data/test'), help='folder with test_<sensor>[_OrigScale]_multiExm1.h5')
    p.add_argument('--device', default='cpu')
    p.add_argument('--baseline', choices=tuple(BASELINES), default='smoc', help='comparison model (default SMOC); c2ms_sp reproduces archived comparisons')
    p.add_argument('--limit', type=int, help='smoke test on the first N scenes (writes metrics_smoke.json, excluded from --summary)')
    p.add_argument('--skip-fr', action='store_true', help='RR only (the run is then marked incomplete)')
    p.add_argument('--summary', action='store_true'); p.add_argument('--figures', action='store_true')
    p.add_argument('--release', action='store_true', help='--figures from the saved predictions Release instead of the weights')
    a = p.parse_args()
    folder = (a.folder or (ROOT / 'benchmarks' / (a.model or ''))).resolve()
    if not (folder / 'model.py').is_file() and not a.summary and not (a.figures and a.release): p.error(f'no adapter: {folder}/model.py')
    if a.summary: summary(folder, a.baseline)
    elif a.figures: figures(a, folder)
    else: evaluate(a, folder)


if __name__ == '__main__':
    main()
