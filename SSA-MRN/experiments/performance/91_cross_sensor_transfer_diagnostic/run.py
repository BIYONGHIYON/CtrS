"""Frozen CAS500 FR diagnostic; source CK weights + experiment-82 calibration, no projection."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seal(args, cfg):
    data = args.repo/'SSA-MRN'/cfg['data']
    meta = json.loads((data.parent/'manifest.json').read_text())
    assert sha(data) == cfg['data_sha256'] == meta['output']['sha256']
    with h5py.File(data, 'r') as f:
        xy = f['ms_yx'][:]
        height, width = meta['geometry']['ms_shape_hw']
        region_end = height//3 - cfg['selection']['guard_ms_pixels']
        candidates = np.flatnonzero(xy[:, 0]+128 <= region_end)
        if len(candidates) < cfg['selection']['count']:
            raise ValueError('Not enough coordinate-only candidates')
        points = xy[candidates].astype(float)/np.array([height/3, width])
        chosen = [0]  # dataset order is lexicographic y,x.
        distance = np.full(len(candidates), np.inf)
        for _ in range(cfg['selection']['count']-1):
            distance = np.minimum(distance, np.sum((points-points[chosen[-1]])**2, axis=1))
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        ids = candidates[chosen].tolist()
        cfg['selected_patch_indices'] = ids
        cfg['selected_ms_yx'] = xy[ids].tolist()
        cfg['candidate_count'] = len(candidates)
        cfg['reserved_region_ms_y_start'] = height//3 + cfg['selection']['guard_ms_pixels']
        assert len(set(ids)) == 20
    write(HERE/'config.json', cfg)
    print(json.dumps({'sealed_patch_ids': ids, 'candidate_count': len(candidates)}), flush=True)


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True, help='Owning source repository with archived weights and prepared data')
    p.add_argument('--output', type=Path)
    p.add_argument('--seal', action='store_true')
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    cfg = json.loads((HERE/'config.json').read_text())
    if args.seal:
        if 'selected_patch_indices' in cfg:
            raise ValueError('Selection is already frozen')
        return seal(args, cfg)
    ids = cfg['selected_patch_indices']
    assert cfg['projection'] is False and cfg['ground_truth'] is False
    out = args.output or args.repo/'SSA-MRN/data/cas500_1_l2g_20260522/diagnostic91'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists() or any(out.glob('predictions_*.h5')):
        raise FileExistsError('Refusing to overwrite diagnostic results')
    root = args.repo/'SSA-MRN'
    sys.path.insert(0, str(root))
    from common.metrics import fr_metrics
    model_path = root/'experiments/performance/90_dataset_matched_internal_resampling/model.py'
    m90 = load_module('cas91_m90', model_path)
    m90.OUT_IMPL['value'] = 'convT'
    data = root/cfg['data']
    assert sha(data) == cfg['data_sha256']
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Metal GPU is unavailable; choose --device cpu explicitly')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    inputs = []
    with h5py.File(data, 'r') as f:
        assert 'gt' not in f and f.attrs['scale_dn'] == 4095
        for patch_id, yx in zip(ids, cfg['selected_ms_yx']):
            assert f['ms_yx'][patch_id].tolist() == yx
            inputs.append({k: f[k][patch_id] for k in ('pan', 'lms', 'ms')})
    records, weights = {}, {}
    for sensor in cfg['sources']:
        checkpoint = root/cfg['checkpoint_paths'][sensor]
        ref_path = root/f'experiments/performance/82_cross_sensor_band_calibration/results/reference_stats/{sensor}.json'
        ref = json.loads(ref_path.read_text())
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ck['seed'] == cfg['seed'] and ck.get('arm') == cfg['arm']
        assert ck.get('sensor') == sensor
        net = m90.Model(cfg['arm'], 4, cfg['seed'], sensor=sensor)
        net.load_state_dict(ck['model'], strict=True)
        net = net.to(device).eval()
        weights[sensor] = dict(path=str(checkpoint), sha256=sha(checkpoint),
                               best_epoch=ck.get('best_epoch'), seed=ck['seed'],
                               calibration_reference_sha256=sha(ref_path),
                               parameters=sum(v.numel() for v in net.parameters()))
        rows = []
        pred_path = out/f'predictions_{sensor}.h5'
        with h5py.File(pred_path, 'x') as dest:
            dest.attrs.update(sensor= sensor, target='CAS500-1', arm='CK', seed=42,
                              calibration='experiment 82 scene mode per patch', projection=False,
                              normalization='float32 target DN / 4095; no output clipping', layout='NCHW')
            ds = dest.create_dataset('prediction', (len(ids), 4, 512, 512), dtype='f4',
                                      chunks=(1,4,512,512), compression='lzf', shuffle=True, fletcher32=True)
            dest.create_dataset('patch_indices', data=np.asarray(ids, dtype='i4'))
            for i, a in enumerate(inputs):
                # Float64 input means are computed on CPU (MPS cannot store float64).
                # These are exactly the t values of experiment 82's scene mode.
                mp = a['pan'].astype('f8').mean()
                mb = a['lms'].astype('f8').mean((1,2))
                t = np.asarray(ref['ratio_of_means'], dtype='f8')*mp/mb
                assert np.isfinite(t).all() and np.all(t > 0)
                t4 = t.astype('f4')[:,None,None]
                arrays = (a['pan'], a['lms']*t4, a['ms']*t4)
                tensors = [torch.from_numpy(np.ascontiguousarray(x))[None].to(device) for x in arrays]
                synchronize(device); started = time.monotonic()
                with torch.inference_mode():
                    prediction = net(*tensors)[0].float().cpu().numpy()/t4
                synchronize(device); seconds = time.monotonic()-started
                assert prediction.shape == (4,512,512) and np.isfinite(prediction).all()
                ds[i] = prediction
                # Public repository's Python QNR implementation; no GT, no RR claims.
                q = fr_metrics(prediction.astype('f8')*4095, a['lms'].astype('f8')*4095,
                               a['ms'].astype('f8')*4095, a['pan'].astype('f8')*4095)
                assert all(np.isfinite(v) for v in q.values())
                rows.append(dict(diagnostic_index=i, patch_index=ids[i], ms_yx=cfg['selected_ms_yx'][i],
                                 input_pan_mean=float(mp), input_band_means=mb.tolist(),
                                 calibration_t=t.tolist(), FR_reference=q,
                                 inference_seconds=seconds, output_min=float(prediction.min()),
                                 output_max=float(prediction.max())))
                write(out/f'progress_{sensor}.json', rows)
                print(sensor, i+1, '/', len(ids), json.dumps(q), f'{seconds:.2f}s', flush=True)
        # Verify every output chunk and its scene order before recording completion.
        with h5py.File(pred_path, 'r') as f:
            assert f['patch_indices'][:].tolist() == ids
            for a in f['prediction']:
                assert np.isfinite(a).all()
        records[sensor] = rows
        weights[sensor]['predictions'] = dict(path=str(pred_path), bytes=pred_path.stat().st_size,
                                               sha256=sha(pred_path), verified_all_chunks=True)
        del net, tensors
        if device == 'mps':
            torch.mps.empty_cache()
    means = {s: {k: float(np.mean([r['FR_reference'][k] for r in rows]))
                 for k in rows[0]['FR_reference']} for s, rows in records.items()}
    source_files = sorted({Path(mod.__file__).resolve() for mod in list(sys.modules.values())
                           if getattr(mod, '__file__', None)
                           and str(Path(mod.__file__).resolve()).startswith(str(root)+'/')
                           and Path(mod.__file__).suffix == '.py'})
    summary = dict(scope='FR diagnostic only; CK+82 without projection, not full SMOC',
                   config=cfg, data=dict(path=str(data), sha256=cfg['data_sha256']), weights=weights,
                   means=means, rows=records, selection='frozen before inference; coordinate-only top-third subset',
                   independent_scenes=1, verdict='성능 비교 아님', winner=None,
                   limitations=['No CAS500 measured bandwise MTF', 'No RR/GT', 'One source scene',
                                'Python FR QNR is reference only; not a selection oracle'],
                   environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                    platform=platform.platform(), device=device),
                   provenance=dict(run_script_sha256=sha(Path(__file__)), config_sha256=sha(HERE/'config.json'),
                                   model_source_sha256=sha(model_path),
                                   imported_sources={str(q.relative_to(root)): dict(bytes=q.stat().st_size,
                                                    sha256=sha(q)) for q in source_files}))
    write(out/'summary.json', summary)
    # Same input-LMS display scale for both source predictions, fixed before looking at outputs.
    preview_ids = cfg['display_indices']
    canvas = Image.new('RGB', (4*512, len(preview_ids)*540), 'white')
    draw = ImageDraw.Draw(canvas)
    with h5py.File(out/'predictions_QB.h5') as qb, h5py.File(out/'predictions_GF2.h5') as gf:
        for row, i in enumerate(preview_ids):
            a = inputs[i]
            rgb_ref = a['lms'][[2,1,0]]
            low = np.percentile(rgb_ref, 2, axis=(1,2))[:,None,None]
            high = np.percentile(rgb_ref, 98, axis=(1,2))[:,None,None]
            def rgb(x):
                x = x[[2,1,0]]
                return Image.fromarray((255*np.clip((x-low)/np.maximum(high-low,1e-9),0,1)).astype('u1').transpose(1,2,0))
            pan = a['pan'][0]; lo,hi=np.percentile(pan,[2,98])
            pictures = [rgb(a['lms']), Image.fromarray((255*np.clip((pan-lo)/(hi-lo),0,1)).astype('u1')).convert('RGB'),
                        rgb(qb['prediction'][i]), rgb(gf['prediction'][i])]
            labels = ['LMS input', 'PAN input', 'QB CK + 82 (no projection)', 'GF2 CK + 82 (no projection)']
            for col, (pic,label) in enumerate(zip(pictures,labels)):
                canvas.paste(pic, (col*512,row*540+28))
                draw.text((col*512+5,row*540+6), f'{label} | patch {ids[i]}', fill='black')
    canvas.save(out/'comparison.png')
    # A smaller contact sheet preserves the same scales and all six predetermined regions.
    canvas.resize((1536, len(preview_ids)*405), Image.Resampling.LANCZOS).save(out/'comparison_contact.png')
    print('COMPLETE', json.dumps(means), str(out), flush=True)


if __name__ == '__main__':
    main()
