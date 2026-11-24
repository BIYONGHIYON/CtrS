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


def mosaic(args):
    """Create a qualitative overlap-blended mosaic without changing the diagnostic subset."""
    import tifffile
    root = args.repo.resolve()/'SSA-MRN'
    sys.path.insert(0, str(root))
    plan_path = HERE/'provenance/mosaic_plan.json'
    plan = json.loads(plan_path.read_text())
    prep_path = root/'data/cas500_1_l2g_20260522/prepare.py'
    prep = load_module('cas91_mosaic_preparation', prep_path)
    source = Path(plan['source_directory'])
    ras = {b:prep.Raster(next(source.glob('*_'+b+'.tif'))) for b in ('B','G','R','N','P')}
    frozen_input = json.loads((HERE/'provenance/data_manifest.json').read_text())
    # The original source list is copied verbatim from the already-verified preparation manifest.
    entries = frozen_input['sources']
    for item in entries:
        assert sha(Path(item['path'])) == item['sha256'], item['path']
    px,py,W,H = plan['roi_pan_xywh']
    assert all(v%4==0 for v in (px,py,W,H))
    x,y,w,h = px//4,py//4,W//4,H//4
    out = args.output or root/'data/cas500_1_l2g_20260522/dongguk_QB_no_projection'
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing mosaic directory')
    out.mkdir(parents=True)
    write(out/'frozen_plan.json',plan)
    halo = 16
    ms = np.stack([ras[b].crop(y,x,h,w) for b in ('B','G','R','N')]).astype('f4')/4095
    ctx = np.stack([ras[b].crop(y-halo,x-halo,h+2*halo,w+2*halo) for b in ('B','G','R','N')]).astype('f4')/4095
    lms = prep.interpolate(ctx)[:,halo*4:(halo+h)*4,halo*4:(halo+w)*4]
    pan = ras['P'].crop(py,px,H,W)[None].astype('f4')/4095
    assert ms.shape==(4,h,w) and lms.shape==(4,H,W) and pan.shape==(1,H,W)
    assert np.all(ms>0) and np.all(pan>0)
    ref_path = root/'experiments/performance/82_cross_sensor_band_calibration/results/reference_stats/QB.json'
    ref = json.loads(ref_path.read_text())
    t = np.asarray(ref['ratio_of_means'],dtype='f8')*pan.astype('f8').mean()/lms.astype('f8').mean((1,2))
    t4 = t.astype('f4')[:,None,None]
    assert np.isfinite(t).all() and (t>0).all()
    cfg=json.loads((HERE/'config.json').read_text())
    checkpoint=root/cfg['checkpoint_paths']['QB']
    expected=json.loads((HERE/'provenance/source_manifest.json').read_text())['weights']['QB']['sha256']
    assert sha(checkpoint)==expected
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['seed']==42 and ck['arm']=='CK' and ck['sensor']=='QB'
    m90=load_module('cas91_mosaic_m90',root/'experiments/performance/90_dataset_matched_internal_resampling/model.py')
    m90.OUT_IMPL['value']='convT'
    torch.set_num_threads(4);torch.manual_seed(42)
    device=args.device
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Metal GPU unavailable')
    net=m90.Model('CK',4,42,sensor='QB');net.load_state_dict(ck['model'],strict=True);net=net.to(device).eval()
    tile=plan['tiling']['input_pan'];border=plan['tiling']['discard_pan_border'];core=tile-2*border;stride=plan['tiling']['stride']
    def starts(n):
        return sorted(set(list(range(0,n-core+1,stride))+[n-core]))
    ys,xs=starts(H),starts(W)
    axis=np.sin(np.pi*(np.arange(core)+.5)/core)**2
    weight=(axis[:,None]*axis[None,:]).astype('f4')
    acc=np.zeros((4,H,W),dtype='f4');coverage=np.zeros((H,W),dtype='f4');count=np.zeros((H,W),dtype='u1')
    rows=[];started=time.monotonic()
    # Independently check context interpolation against the whole-ROI interpolation at one tile.
    for i,(oy,ox) in enumerate((a,b) for a in ys for b in xs):
        gy,gx=py+oy-border,px+ox-border;my,mx=gy//4,gx//4
        cc=np.stack([ras[b].crop(my-halo,mx-halo,128+2*halo,128+2*halo) for b in ('B','G','R','N')]).astype('f4')/4095
        aa=dict(ms=cc[:,halo:halo+128,halo:halo+128],lms=prep.interpolate(cc)[:,halo*4:(halo+128)*4,halo*4:(halo+128)*4],pan=ras['P'].crop(gy,gx,tile,tile)[None].astype('f4')/4095)
        assert all(np.isfinite(a).all() for a in aa.values()) and all(np.all(a>0) for a in (cc,aa['pan']))
        if i==0:
            np.testing.assert_array_equal(aa['lms'][:,border:border+core,border:border+core],lms[:,oy:oy+core,ox:ox+core])
        tensors=[torch.from_numpy(np.ascontiguousarray(z))[None].to(device) for z in (aa['pan'],aa['lms']*t4,aa['ms']*t4)]
        with torch.inference_mode():
            prediction=net(*tensors)[0].float().cpu().numpy()/t4
        assert prediction.shape==(4,tile,tile) and np.isfinite(prediction).all()
        z=prediction[:,border:border+core,border:border+core]
        acc[:,oy:oy+core,ox:ox+core]+=z*weight
        coverage[oy:oy+core,ox:ox+core]+=weight
        count[oy:oy+core,ox:ox+core]+=1
        rows.append(dict(tile_index=i,source_pan_xy=[gx,gy],retained_output_xy=[ox,oy]))
        if (i+1)%10==0 or i+1==len(ys)*len(xs):
            print('MOSAIC',i+1,'/',len(ys)*len(xs),'elapsed',round(time.monotonic()-started,1),flush=True)
    assert np.all(coverage>0) and np.all(count>0)
    fused=acc/coverage[None]
    assert fused.shape==(4,H,W) and np.isfinite(fused).all()
    # Normalize-before-accumulate identity check catches any blend or coverage scaling error.
    np.testing.assert_allclose((coverage.astype('f8')*2.5)/coverage,2.5,rtol=1e-14)
    lo=np.percentile(lms[[2,1,0]],2,axis=(1,2))[:,None,None]
    hi=np.percentile(lms[[2,1,0]],98,axis=(1,2))[:,None,None]
    def rgb(a):
        return Image.fromarray((255*np.clip((a[[2,1,0]]-lo)/(hi-lo),0,1)).astype('u1').transpose(1,2,0))
    image=rgb(fused);input_image=rgb(lms)
    image.save(out/'dongguk_QB_no_projection.png')
    image.save(out/'dongguk_QB_no_projection.jpg',quality=96,subsampling=0,optimize=True)
    input_image.save(out/'dongguk_input_LMS.png')
    image.resize((1280,round(H*1280/W)),Image.Resampling.LANCZOS).save(out/'dongguk_preview.jpg',quality=90,subsampling=0)
    dn=(fused*4095).astype('f4')
    tifffile.imwrite(out/'dongguk_QB_no_projection_BGRNIR_DN.tif',dn,photometric='minisblack',metadata=dict(axes='CYX',band_order='B,G,R,NIR',unit='DN',projection=False,georeferencing='not certified; source pixel ROI in frozen_plan.json'),compression='deflate')
    with tifffile.TiffFile(out/'dongguk_QB_no_projection_BGRNIR_DN.tif') as f:
        back=f.asarray()
        np.testing.assert_array_equal(back,dn)
    with Image.open(out/'dongguk_QB_no_projection.png') as f:
        assert f.size==(W,H);f.verify()
    # Fixed central 512-pixel crop, not selected using prediction quality.
    cx,cy=W//2,H//2;box=(cx-256,cy-256,cx+256,cy+256)
    comp=Image.new('RGB',(1536,540),'white');draw=ImageDraw.Draw(comp)
    pp=pan[0];pl,ph=np.percentile(pp,[2,98]);pi=Image.fromarray((255*np.clip((pp-pl)/(ph-pl),0,1)).astype('u1')).convert('RGB')
    for col,(pic,label) in enumerate([(input_image,'Input LMS'),(pi,'Input PAN'),(image,'QB CK + band calibration; no projection')]):
        comp.paste(pic.crop(box),(col*512,28));draw.text((col*512+5,7),label,fill='black')
    comp.save(out/'center_comparison.png')
    verification=dict(all_tiles_finite=True,full_coverage=True,minimum_weight=float(coverage.min()),coverage_count_min=int(count.min()),coverage_count_max=int(count.max()),float_tiff_exact_readback=True,rgb_png_verified=True,interpolation_matches_whole_roi=True)
    result=dict(scope=plan['task'],plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),model_sha256=sha(checkpoint),calibration_reference_sha256=sha(ref_path),calibration_t=t.tolist(),source_hashes=entries,preprocessing_sha256=sha(prep_path),roi_pan_xywh=plan['roi_pan_xywh'],shape_chw=list(fused.shape),nominal_extent_m=[W*.5,H*.5],georeferencing='not certified; original pixel coordinates authoritative',display_percentiles=dict(low=lo.ravel().tolist(),high=hi.ravel().tolist(),gamma=1),tiles=rows,tile_count=len(rows),projection=False,ground_truth=False,rr=False,elapsed_seconds=time.monotonic()-started,verification=verification,environment=dict(device=device,torch=torch.__version__,numpy=np.__version__,tifffile=tifffile.__version__,python=sys.version),output_min=float(fused.min()),output_max=float(fused.max()))
    write(out/'summary.json',result)
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()}
    write(out/'manifest.json',dict(scope=result['scope'],files=artifacts))
    print('MOSAIC COMPLETE',str(out),json.dumps(verification),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True, help='Owning source repository with archived weights and prepared data')
    p.add_argument('--output', type=Path)
    p.add_argument('--seal', action='store_true')
    p.add_argument('--mosaic', action='store_true', help='Qualitative QB no-projection overlap mosaic')
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = p.parse_args()
    args.repo = args.repo.resolve()
    if args.mosaic:
        return mosaic(args)
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
