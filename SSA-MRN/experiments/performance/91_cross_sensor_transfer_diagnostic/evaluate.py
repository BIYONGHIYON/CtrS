"""Apply a predeclared assumed CAS500 MS operator to both archived diagnostic predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def make_projector(root, plan):
    p79 = module('cas91_projection79', root/'experiments/performance/79_deep_narrow_branch_projection/prepare.py')
    kernels = np.stack([p79.gen_mtf(g, plan['ratio'], plan['kernel_size']) for g in plan['ms_gnyq']])[:,None]
    class AssumedProjector(p79.Projector):
        def __init__(self):
            self.D = p79.make_D(torch.from_numpy(kernels), phase=plan['phase_yx'][0], ratio=plan['ratio'])
            self.margin = plan['ms_margin']
            self.tol = plan['relative_tolerance']
            self.max_iter = plan['max_iterations']
    assert plan['phase_yx'][0] == plan['phase_yx'][1]
    return AssumedProjector(), kernels, p79


def preflight(root, plan):
    torch.set_num_threads(4)
    projector, k, p79 = make_projector(root, plan)
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1,4,64,64,generator=g,dtype=torch.float64)
    p54 = module('cas91_projection54', root/'experiments/performance/54_multi_sensor_scale_mixed_training/prepare.py')
    direct = p54._blur(x, torch.from_numpy(k))[...,2::4,2::4]
    fft = projector.D(x)
    maximum = float((direct-fft).abs().max())
    assert maximum < 1e-12
    v = torch.randn(fft.shape, generator=g, dtype=torch.float64)
    lhs, rhs = float((projector.D(x)*v).sum()), float((x*projector.Dt(v,x.shape)).sum())
    error = abs(lhs-rhs)/max(abs(lhs),abs(rhs),1.)
    assert error < 1e-12
    projected, it = projector(x, projector.D(x))
    fixed_error = float((projected-x).abs().max())
    assert fixed_error < 1e-12
    return dict(fft_vs_direct_max_abs=maximum, adjoint_relative_error=error,
                zero_residual_max_change=fixed_error, passed=True,
                kernel_sha256=hashlib.sha256(k.tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    root = args.repo.resolve()/'SSA-MRN'
    sys.path.insert(0, str(root))
    from common.metrics import fr_metrics
    cfg = json.loads((HERE/'config.json').read_text())
    plan_path = HERE/'provenance/projection_plan.json'
    plan = json.loads(plan_path.read_text())
    checks = preflight(root,plan)
    print('PREFLIGHT',json.dumps(checks),flush=True)
    if args.preflight_only:
        return
    baseline = root/'data/cas500_1_l2g_20260522/diagnostic91'
    old_path = baseline/'summary.json'
    old = json.loads(old_path.read_text())
    assert old['config'] == cfg
    assert old['provenance']['config_sha256'] == sha(HERE/'config.json')
    data = root/cfg['data']
    assert sha(data) == cfg['data_sha256']
    out = args.output or baseline/'assumed_mtf026_k10'
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing projection comparison')
    out.mkdir(parents=True)
    write(out/'frozen_plan.json', plan)
    write(out/'preflight.json', checks)
    projector, kernels, p79 = make_projector(root,plan)
    ids = cfg['selected_patch_indices']
    def residual(y,ms):
        e = projector.D(y)-ms
        m = projector.margin
        return float(e[...,m:-m,m:-m].square().mean().sqrt()*4095.)
    records, prediction_files = {}, {}
    with h5py.File(data,'r') as f:
        for sensor in plan['sources']:
            src_path = baseline/f'predictions_{sensor}.h5'
            assert sha(src_path) == old['weights'][sensor]['predictions']['sha256']
            dest_path = out/f'predictions_{sensor}_assumed_P.h5'
            rows = []
            with h5py.File(src_path,'r') as src, h5py.File(dest_path,'x') as dest:
                assert src['patch_indices'][:].tolist() == ids
                dest.attrs.update(source=sensor,target='CAS500-1',seed=42,
                                  projection='assumed MS GNyq .26 all bands, CG K10',
                                  phase_yx=[2,2],ms_margin=5,measured_mtf=False,
                                  normalization='float32 DN/4095, no clipping')
                dest.create_dataset('patch_indices',data=np.asarray(ids,dtype='i4'))
                ds = dest.create_dataset('prediction',(len(ids),4,512,512),dtype='f4',
                                         chunks=(1,4,512,512),compression='lzf',shuffle=True,fletcher32=True)
                for i,pid in enumerate(ids):
                    assert f['ms_yx'][pid].tolist() == cfg['selected_ms_yx'][i]
                    a = {key:f[key][pid].astype('f8') for key in ('pan','ms','lms')}
                    y = torch.from_numpy(src['prediction'][i].astype('f8'))[None]
                    ms = torch.from_numpy(a['ms'])[None]
                    before = residual(y,ms)
                    start = time.monotonic()
                    with torch.enable_grad():
                        projected,it = projector(y,ms)
                    elapsed = time.monotonic()-start
                    z = projected.detach()[0].numpy().astype('f4')
                    assert z.shape == (4,512,512) and np.isfinite(z).all()
                    after = residual(torch.from_numpy(z.astype('f8'))[None],ms)
                    assert after < before
                    q = fr_metrics(z.astype('f8')*4095,a['lms']*4095,a['ms']*4095,a['pan']*4095)
                    assert all(np.isfinite(v) for v in q.values())
                    ds[i] = z
                    row = dict(diagnostic_index=i,patch_index=pid,ms_yx=cfg['selected_ms_yx'][i],
                               before_FR_reference=old['rows'][sensor][i]['FR_reference'],after_FR_reference=q,
                               assumed_observation_rmse_dn_before=before,
                               assumed_observation_rmse_dn_after=after,
                               cg_iterations=it,projection_seconds=elapsed,
                               output_min=float(z.min()),output_max=float(z.max()))
                    rows.append(row)
                    print(sensor,i+1,'/',len(ids),json.dumps(q),f'RMSE {before:.5f}->{after:.5f} DN',flush=True)
            with h5py.File(dest_path,'r') as d:
                assert d['patch_indices'][:].tolist() == ids
                for x in d['prediction']:
                    assert np.isfinite(x).all()
            records[sensor] = rows
            prediction_files[sensor] = dict(path=str(dest_path),bytes=dest_path.stat().st_size,
                                            sha256=sha(dest_path),verified_all_chunks=True)
    means = {s:dict(before=old['means'][s],
                    after={k:float(np.mean([r['after_FR_reference'][k] for r in rows]))
                           for k in rows[0]['after_FR_reference']},
                    assumed_observation_rmse_dn_before=float(np.mean([r['assumed_observation_rmse_dn_before'] for r in rows])),
                    assumed_observation_rmse_dn_after=float(np.mean([r['assumed_observation_rmse_dn_after'] for r in rows])))
             for s,rows in records.items()}
    result = dict(scope='Assumed-MTF projection reference only, not a measured CAS500 observation model',
                  plan=plan,plan_sha256=sha(plan_path),checks=checks,means=means,rows=records,
                  predictions=prediction_files,independent_scenes=1,seed=42,
                  source_baseline=dict(path=str(old_path),sha256=sha(old_path)),
                  source_data=dict(path=str(data),sha256=cfg['data_sha256']),
                  sources={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [
                      root/'experiments/performance/79_deep_narrow_branch_projection/prepare.py',
                      root/'experiments/performance/54_multi_sensor_scale_mixed_training/prepare.py',root/'common/metrics.py']},
                  script_sha256=sha(Path(__file__)),dtype='CPU float64, stored float32',
                  physical_accuracy_verified=False,ground_truth=False,rr=False)
    write(out/'summary.json',result)
    canvas = Image.new('RGB',(5*384,len(plan['display_indices'])*408),'white')
    draw = ImageDraw.Draw(canvas)
    with h5py.File(data) as f,h5py.File(baseline/'predictions_QB.h5') as q0,h5py.File(baseline/'predictions_GF2.h5') as g0,h5py.File(out/'predictions_QB_assumed_P.h5') as q1,h5py.File(out/'predictions_GF2_assumed_P.h5') as g1:
        for row,i in enumerate(plan['display_indices']):
            lms=f['lms'][ids[i]];ref=lms[[2,1,0]]
            lo=np.percentile(ref,2,axis=(1,2))[:,None,None];hi=np.percentile(ref,98,axis=(1,2))[:,None,None]
            arrays=[lms,q0['prediction'][i],q1['prediction'][i],g0['prediction'][i],g1['prediction'][i]]
            labels=['LMS input','QB no P','QB assumed P','GF2 no P','GF2 assumed P']
            for col,(a,label) in enumerate(zip(arrays,labels)):
                rgb=(255*np.clip((a[[2,1,0]]-lo)/np.maximum(hi-lo,1e-9),0,1)).astype('u1').transpose(1,2,0)
                im=Image.fromarray(rgb).resize((384,384),Image.Resampling.LANCZOS)
                canvas.paste(im,(384*col,row*408+24))
                draw.text((384*col+4,row*408+5),f'{label} | patch {ids[i]}',fill='black')
    canvas.save(out/'comparison.png')
    canvas.save(out/'comparison_review.jpg',quality=82,optimize=True)
    canvas.crop((0,0,1920,408)).save(out/'comparison_first.png')
    print('COMPLETE',json.dumps(means),str(out),flush=True)


if __name__ == '__main__':
    main()
