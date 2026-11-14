"""Export RR test inputs, ground truth and model predictions for the result viewer (viewer/index.html).

For every dataset and model in registry.json this writes viewer/data/<dataset>/s<scene>/<array>.i16 (int16, little endian,
CHW, value = DN x 8, so the stored precision is 1/8 DN) and viewer/data/index.json with the whole-scene RR metrics
computed by common/metrics.rr_metrics on the float predictions (PSNR, SAM, ERGAS, SCC, Q2n). Each dataset also gets
thumbs.jpg, one row of ground-truth thumbnails for the scene picker.

A model is either
- built from code: "module" (path below SSA-MRN), "factory" and "args" ("{bands}" is replaced by the band count) and one
  "checkpoints" entry per dataset (torch file with the state dict under "state_key", default "model"; "{dataset}" in args is replaced by the
  dataset id and "kwargs_by_dataset" adds keyword arguments for one dataset); inputs are
  (pan, lms, ms) divided by the dataset "scale" and the output is multiplied back, or
- given as precomputed predictions: "predictions": {"<dataset>": "<h5 path>"} with dataset "pred" of shape (N, C, H, W) in DN.
Datasets without an entry for a model are skipped for that model. Existing results of models that are not exported
again are kept in index.json.

Usage: python SSA-MRN/viewer/export.py [--models ssa_mrn c2ms_sp c2ms_sp_lite] [--datasets QB GF2] [--device mps|cuda|cpu]
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics  # noqa: E402

FACTOR = 8  # stored int16 = round(DN * FACTOR)
THUMB = 112  # scene picker thumbnail size (pixels)
METRICS = ('PSNR', 'SAM', 'ERGAS', 'SCC', 'Q2n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def save(path, array):
    q = np.rint(np.asarray(array, dtype=np.float64) * FACTOR)
    clipped = int(np.sum((q < -32768) | (q > 32767)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.clip(q, -32768, 32767).astype('<i2').tofile(path)
    return clipped


def thumbnails(path, scenes, bands):
    """One row of ground-truth thumbnails (GT 1-99% stretch per scene, first composite of the dataset)."""
    sheet = Image.new('RGB', (THUMB * len(scenes), THUMB))
    for i, s in enumerate(scenes):
        x = s['gt'][bands].transpose(1, 2, 0); lo, hi = np.percentile(x, [1, 99], axis=(0, 1))
        tile = Image.fromarray((np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype('uint8'))
        sheet.paste(tile.resize((THUMB, THUMB), Image.Resampling.LANCZOS), (i * THUMB, 0))
    sheet.save(path, quality=88)


def build(model, dataset, device):
    spec = importlib.util.spec_from_file_location(f'viewer_{model["id"]}', ROOT / model['module'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = [dataset['bands'] if a == '{bands}' else dataset['id'] if a == '{dataset}' else a for a in model.get('args', [])]
    kwargs = {**model.get('kwargs', {}), **model.get('kwargs_by_dataset', {}).get(dataset['id'], {})}
    net = getattr(module, model['factory'])(*args, **kwargs)
    path = ROOT / model['checkpoints'][dataset['id']]
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck[model.get('state_key', 'model')] if isinstance(ck, dict) and model.get('state_key', 'model') in ck else ck
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), dict(checkpoint=model['checkpoints'][dataset['id']], checkpoint_sha256=sha(path), best_epoch=ck.get('best_epoch') if isinstance(ck, dict) else None)


def predictor(model, dataset, device):
    if 'predictions' in model:
        path = ROOT / model['predictions'][dataset['id']]
        with h5py.File(path, 'r') as f:
            pred = np.asarray(f['pred'], dtype=np.float64)
        return (lambda i, a: pred[i]), dict(predictions=model['predictions'][dataset['id']], predictions_sha256=sha(path))
    net, info = build(model, dataset, device)
    scale = float(dataset['scale'])

    def run(i, a):
        x = [torch.from_numpy((a[k] / scale).astype('float32')).unsqueeze(0).to(device) for k in ('pan', 'lms', 'ms')]
        with torch.no_grad():
            out = net(*x)[0].float().cpu().numpy().astype(np.float64) * scale
        if not np.isfinite(out).all(): raise FloatingPointError(f'nonfinite prediction {model["id"]} {dataset["id"]} scene {i}')
        return out
    return run, info


def available(model, dataset):
    return dataset['id'] in model.get('predictions', model.get('checkpoints', {}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'data', help='folder containing the PanCollection RR test H5 files')
    p.add_argument('--out', type=Path, default=HERE / 'data')
    p.add_argument('--models', nargs='*', help='model ids to export (default: all)')
    p.add_argument('--datasets', nargs='*', help='dataset ids to export (default: all)')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    reg = json.loads((HERE / 'registry.json').read_text(encoding='utf-8'))
    index_path = a.out / 'index.json'
    old = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    old_ds = {d['id']: d for d in old.get('datasets', [])}
    chosen = set(a.models or [m['id'] for m in reg['models']])
    datasets = []
    for ds in reg['datasets']:
        prev = old_ds.get(ds['id'], {})
        entry = {k: v for k, v in ds.items() if k != 'rr_file'}
        entry['results'] = {k: v for k, v in prev.get('results', {}).items() if k not in chosen and any(m['id'] == k for m in reg['models'])}
        for key in ('scenes', 'size', 'ms_size', 'data_sha256', 'thumbnails'):
            if key in prev: entry[key] = prev[key]
        datasets.append(entry)
        if a.datasets and ds['id'] not in a.datasets: continue
        found = list(a.data.rglob(ds['rr_file']))
        if len(found) != 1: raise FileNotFoundError(f'{ds["rr_file"]} below {a.data}: {found}')
        with h5py.File(found[0], 'r') as f:
            scenes = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(len(f['gt']))]
        entry.update(scenes=len(scenes), size=int(scenes[0]['gt'].shape[-1]), ms_size=int(scenes[0]['ms'].shape[-1]), data_sha256=sha(found[0]))
        out = a.out / ds['id']
        for i, s in enumerate(scenes):
            for k in ('gt', 'pan', 'ms'): save(out / f's{i:02}/{k}.i16', s[k])
        thumbnails(out / 'thumbs.jpg', scenes, next(iter(ds['composites'].values()))); entry['thumbnails'] = 'thumbs.jpg'
        for model in reg['models']:
            if model['id'] not in chosen or not available(model, ds): continue
            t = time.time(); run, info = predictor(model, ds, a.device); rows, clipped = [], 0
            for i, s in enumerate(scenes):
                pred = run(i, s)
                if pred.shape != s['gt'].shape: raise ValueError(f'{model["id"]} {ds["id"]} scene {i}: shape {pred.shape} != {s["gt"].shape}')
                clipped += save(out / f's{i:02}/{model["id"]}.i16', pred)
                m = rr_metrics(s['gt'], pred); rows.append({k: float(m[k]) for k in METRICS})
            mean = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
            entry['results'][model['id']] = dict(scene_metrics=rows, mean=mean, clipped_values=clipped, device=a.device, **info)
            print(f'{ds["id"]} {model["id"]}: PSNR {mean["PSNR"]:.4f} SAM {mean["SAM"]:.4f} ERGAS {mean["ERGAS"]:.4f} ({time.time() - t:.0f}s, clipped {clipped})', flush=True)
    models = [{k: v for k, v in m.items() if k not in ('module', 'factory', 'args', 'kwargs', 'kwargs_by_dataset', 'state_key')} for m in reg['models']]
    index = dict(generated=time.strftime('%Y-%m-%d %H:%M'), storage=dict(dtype='int16 little endian', layout='CHW', factor=FACTOR),
                 metrics=dict(source='common/metrics.rr_metrics, float predictions', psnr_peak=2047), datasets=datasets, models=models)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('wrote', index_path)


if __name__ == '__main__':
    main()
