"""Reproduce six saved U2Net scenes and render C2MS-SP seed42 comparisons. No training."""
import argparse, hashlib, importlib.util, json, shutil, sys, time, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCENES=(0,1,3,7,14,17)
BASE_SHA='d9d9d8c67bca9f48f8e6027825041f76a8c5a7e987a26a68287e34fcbd351541'
U2_SHA='e5ef136cdf95cf3cd4ad91bc7d2cea16fba9d8070d82eb1165822d458054bef0'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def find_rr(root,name):
    files=[p for p in root.rglob('*.h5') if (p.name==name or p.name.endswith('__'+name)) and not p.name.startswith('._') and '__._' not in p.name]
    if len(files)!=1:raise ValueError(f'{name}: expected one file, found {len(files)}; narrow --input if duplicated')
    return files[0]

def main():
    a=argparse.ArgumentParser();a.add_argument('--input',type=Path,default=Path('/kaggle/input'))
    a.add_argument('--output',type=Path,default=Path('/kaggle/working/U2Net_C2MSSP_images_v1'))
    a.add_argument('--sensor',choices=['both','WV3','WV2'],default='both');a.add_argument('--device',default='cuda:0')
    args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'package_manifest.json').read_text(encoding='utf-8'))
    for name,v in manifest['files'].items():
        if sha(HERE/name)!=v['sha256']:raise ValueError('Package file changed: '+name)
    sys.path.insert(0,str(HERE/'SSA-MRN'))
    import torch,h5py,numpy as np
    from common import baseline_models as B,result_images as R,evaluation_protocol as P
    from common.metrics import rr_metrics
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Enable GPU, or explicitly choose --device cpu')
    # Original renderer, with a bundled OFL font for Korean titles.
    from PIL import ImageFont
    font=ImageFont.truetype(str(HERE/'fonts/NanumGothic-Regular.ttf'),15)
    spec=importlib.util.spec_from_file_location('author_u2net_adapter',HERE/'u2net/model.py')
    U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
    candidate=U.build(8,'WV3','author',None);U.load(candidate,HERE/'u2net/200000_iter.pth');candidate.to(args.device).eval()
    pkg_sha=sha(HERE/'package_manifest.json')
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             baseline='C2MS-SP seed42, K10 projection; WV2 includes experiment82 scene band calibration',
             candidate='U2Net official author WV3 weights, seed unknown; 64px tiles, halo4, clipping[0,1]',
             package_sha256=pkg_sha,baseline_weight_sha256=BASE_SHA,u2net_weight_sha256=U2_SHA,
             note='Images use one baseline seed; existing aggregate table uses C2MS-SP 3seed mean. No new training or test selection.')
    write(args.output/'environment.json',env)
    try:
        for sensor in (('WV3','WV2') if args.sensor=='both' else (args.sensor,)):
            folder=args.output/sensor;folder.mkdir(exist_ok=True)
            with zipfile.ZipFile(HERE/'prior_results.zip') as z:
                old=json.loads(z.read(f'u2net/results/author_{sensor}/metrics.json'))
            if old['checkpoint']['sha256']!=U2_SHA or not old['complete']:raise ValueError('Prior result weight/completeness mismatch')
            name=f'test_{sensor.lower()}_multiExm1.h5';path=find_rr(args.input,name)
            data_sha=sha(path)
            if data_sha!=old['data'][name]['sha256']:raise ValueError('RR data does not match supplied results: '+sensor)
            old_by_scene={r['scene']:r for r in old['rr']['per_scene']}
            baseline=B.build('c2ms_sp',sensor,42,device=args.device)
            names=['c2ms_sp_s42','u2net_author'];labels=['C2MS-SP (seed42)','U2Net (author, seed 미상)']
            with h5py.File(path,'r') as f:
                if f['gt'].shape!=(20,8,256,256):raise ValueError('Expected 20x8x256x256 RR GT')
                for i in SCENES:
                    marker=folder/f'scene_{i:02}.json'
                    if marker.exists():
                        done=json.loads(marker.read_text(encoding='utf-8'))
                        if done['package_sha256']!=pkg_sha or done['data_sha256']!=data_sha:raise ValueError('Resume provenance changed; choose a new output folder')
                        if all((folder/n).is_file() and sha(folder/n)==v for n,v in done['images'].items()):
                            print(f'{sensor} scene {i}: saved, skipped',flush=True);continue
                    t=time.perf_counter();scene={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')}
                    pred={names[0]:P.predict(baseline,scene['pan'],scene['lms'],scene['ms'],args.device),
                          names[1]:P.predict(candidate,scene['pan'],scene['lms'],scene['ms'],args.device)}
                    metrics={n:rr_metrics(scene['gt'],v) for n,v in pred.items()}
                    mismatch=abs(metrics[names[1]]['PSNR']-old_by_scene[i]['PSNR'])
                    if mismatch>1e-3:raise ValueError(f'U2Net scene {i} PSNR differs from saved result by {mismatch:.6g}dB; stop before image selection')
                    title=f'{sensor} RR 장면 {i} · C2MS-SP seed42 / U2Net author(seed 미상) · PSNR '+ ' / '.join(f'{metrics[n]["PSNR"]:.3f}' for n in names)+' dB'
                    full=folder/f'full_scene_{i:02}.png';zoom=folder/f'compare_scene_{i:02}.png'
                    R._full_multi(scene,pred,names,title,full,font,labels,[4,2,1])
                    crop=R._zoom_multi(scene,pred,names,title,zoom,font,labels,[4,2,1],chooser=1)
                    write(marker,dict(sensor=sensor,scene=i,baseline_seed=42,u2net_seed=None,package_sha256=pkg_sha,data_sha256=data_sha,
                        metrics=metrics,rr_delta_u2net_minus_baseline={k:metrics[names[1]][k]-metrics[names[0]][k] for k in ('PSNR','SAM','ERGAS','SCC','Q2n')},
                        prior_u2net_psnr=old_by_scene[i]['PSNR'],reproduction_psnr_abs_error=mismatch,seconds=time.perf_counter()-t,
                        zoom=crop,selection='Fixed scenes before inference; zoom chosen in favour of U2Net using GT error reduction.',
                        images={full.name:sha(full),zoom.name:sha(zoom)}))
                    print(f'{sensor} scene {i}: 2 images saved, {time.perf_counter()-t:.1f}s, U2Net reproduction ΔPSNR={mismatch:.2g}',flush=True)
            rows=[json.loads((folder/f'scene_{i:02}.json').read_text(encoding='utf-8')) for i in SCENES]
            write(folder/'comparison_summary.json',dict(sensor=sensor,complete=True,scenes=list(SCENES),images=12,per_scene=rows,
                  note='Six-scene visualization only, not a replacement for the prior full20-scene evaluation. Zoom favours U2Net.'))
            del baseline
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:
        # Preserve a self-contained source/weights package, plus both prior numerical evaluations.
        repro=args.output/'reproduction';repro.mkdir(exist_ok=True)
        for p in HERE.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                dst=repro/p.relative_to(HERE);dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():shutil.copy2(p,dst)
        hashes={p.relative_to(args.output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        write(args.output/'artifact_manifest.json',dict(files=hashes))
        archive=args.output.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in args.output.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(args.output))
        print('Download:',archive,'Check sensor comparison_summary.json for completion.',flush=True)

if __name__=='__main__':main()
