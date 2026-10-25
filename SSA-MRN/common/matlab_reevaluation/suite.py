"""Reevaluate retained experiment 53/54 best weights. Never imports a training entrypoint."""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
SSA=HERE.parents[1]
REPO=SSA.parent
PERF=SSA/'experiments/performance'
sys.path.insert(0,str(SSA))
EXP={53:PERF/'53_scale_mixed_training',54:PERF/'54_multi_sensor_scale_mixed_training',
     43:PERF/'43_normalized_parallel_capacity',48:PERF/'48_parallel_capacity_observation_refinement',
     49:PERF/'49_observation_refinement_seed_repeat'}
DATA_REV='dbb764fb81aed9b9d75ce0a8339ea205edbdd165'
WV3_RR_REV='b39e5daf0a322105977a7d4ce53ca0c4806defed'
DLPAN_REV='a34f884af88967580e3ef4f61a89e12fdfe4cda5'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def cases(experiments=(53,54)):
    result=[]
    for exp in experiments:
        if exp==53:
            for arm in ('C0','C2','F1','C2MS','F1MS'):
                for seed in ((42,) if arm=='F1MS' else (42,43,44)):
                    result.append(dict(experiment=53,sensor='QB',trained_sensor='QB',arm=arm,seed=seed))
        elif exp==54:
            for sensor in ('GF2','WV3','WV2'):
                for arm in ('C0','C2MS'):
                    for seed in (42,43,44):
                        result.append(dict(experiment=54,sensor=sensor,trained_sensor='WV3' if sensor=='WV2' else sensor,arm=arm,seed=seed))
        else:raise ValueError('Only existing experiments 53 and 54 are in scope')
    for row in result:row['id']=f"e{row['experiment']}_{row['sensor']}_{row['arm']}_s{row['seed']}"
    return result


def checkpoint(case):
    arm,sensor,seed=case['arm'],case['trained_sensor'],case['seed']
    origin=54 if case['experiment']==54 else {'C0':43,'C2':43,'F1':49,'C2MS':53,'F1MS':53}[arm]
    folder=EXP[origin]
    path=folder/f'checkpoints/{arm}_{sensor}_s{seed}/best.pt'
    expected=read(folder/'manifest.json')['files'][path.relative_to(folder).as_posix()]['sha256']
    return path,expected


def network(case):
    import torch
    if case['experiment']==54:
        cls=module('retained54_model',EXP[54]/'model.py').Model
        return cls(case['arm'],8 if case['sensor'] in ('WV3','WV2') else 4,case['seed'])
    if case['arm'] in ('C0','C2'):
        cls=module('retained43_model',EXP[43]/'model.py').Model
    elif case['arm']=='F1':
        cls=module('retained48_model',EXP[48]/'model.py').Model
    else:cls=module('retained53_model',EXP[53]/'model.py').Model
    return cls(case['arm'],case['seed'])


def data_files(sensor):
    cfg=read(EXP[53 if sensor=='QB' else 54]/'config.json')
    expected=cfg['data_sha256'] if sensor=='QB' else (cfg['transfer_data_sha256'][sensor] if sensor=='WV2' else cfg['data_sha256'][sensor])
    result={}
    for protocol in ('RR','FR'):
        name=f"test_{sensor.lower()}{'_OrigScale' if protocol=='FR' else ''}_multiExm1.h5"
        revision=WV3_RR_REV if sensor=='WV3' and protocol=='RR' else DATA_REV
        result[protocol]=dict(name=name,sha256=expected[protocol],revision=revision,
            url=f'https://huggingface.co/datasets/elsting/PanCollection/resolve/{revision}/test_data/{name}?download=true')
    return result


def download(spec,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file() and sha(destination)==spec['sha256']:return destination
    temp=destination.with_suffix('.download')
    error=None
    for attempt in range(3):
        try:
            request=urllib.request.Request(spec['url']+f'&attempt={attempt}&t={time.time_ns()}',headers={'User-Agent':'CtrS-MATLAB-Reevaluation/1'})
            with urllib.request.urlopen(request,timeout=120) as r,temp.open('wb') as f:
                while True:
                    data=r.read(4*1024**2)
                    if not data:break
                    f.write(data)
            actual=sha(temp)
            if actual!=spec['sha256']:raise ValueError(f'Data SHA mismatch for {spec["name"]}: {actual}')
            os.replace(temp,destination);return destination
        except Exception as exc:
            error=exc
            if attempt<2:time.sleep(3)
    raise RuntimeError(f'Cannot acquire exact original data {spec["name"]}: {error}')


def old_rows(case):
    report=read(EXP[case['experiment']]/'results/evaluation_v2_3.json')
    # Keep the raw previous report in provenance. Its layout is intentionally not guessed.
    return report


def export(case,out,data_root,threads=2,limit=None):
    import h5py
    import numpy as np
    import torch
    from scipy.io import savemat
    torch.set_num_threads(threads)
    torch.backends.mkldnn.enabled=True
    torch.use_deterministic_algorithms(True)
    model=network(case).cpu().eval()
    from common.metrics import rr_metrics,fr_metrics
    wald_mod=module('retained54_wald',EXP[54]/'prepare.py')
    wald=wald_mod.Wald(case['sensor'],'cpu')
    peak=1023 if case['sensor']=='GF2' else 2047
    bits=10 if case['sensor']=='GF2' else 11
    cp,expected=checkpoint(case)
    if sha(cp)!=expected:raise ValueError('Retained best hash mismatch: '+str(cp))
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    output=Path(out).resolve();output.mkdir(parents=True,exist_ok=True)
    predictions=output/'predictions';predictions.mkdir(exist_ok=True)
    paths={k:download(spec,Path(data_root)/spec['name']) for k,spec in data_files(case['sensor']).items()}
    records,files=[],[]
    started=time.monotonic()

    def predict(a):
        tensors=[torch.from_numpy((a[k]/peak).astype('float32')).unsqueeze(0) for k in ('pan','lms','ms')]
        with torch.inference_mode():y=model(*tensors)[0].float().numpy().astype('float64')*peak
        if not np.isfinite(y).all():raise FloatingPointError('Nonfinite prediction')
        return y

    def store(protocol,index,a,y,metrics,extra=None):
        path=predictions/f'{protocol}_{index:02}.mat'
        content=dict(fused=y,lms=a['lms'],ms=a['ms'],pan=a['pan'])
        if 'gt' in a:content['reference']=a['gt']
        savemat(path,{k:v.transpose(1,2,0) for k,v in content.items()},do_compression=False)
        files.append(dict(protocol=protocol,scene=index,file='predictions/'+path.name,sha256=sha(path)))
        records.append(dict(protocol=protocol,scene=index,python_legacy=metrics,**(extra or {})))

    with h5py.File(paths['RR'],'r') as rr,h5py.File(paths['FR'],'r') as fr:
        if len(rr['pan'])!=20 or len(fr['pan'])!=20:raise ValueError('Full original RR20/FR20 required')
        for f,required in ((rr,('gt','pan','lms','ms')),(fr,('pan','lms','ms'))):
            if not set(required)<=set(f):raise ValueError('Missing H5 keys')
        for i in range(limit or 20):
            t=time.monotonic()
            r={k:np.asarray(rr[k][i],dtype='float64') for k in ('gt','pan','lms','ms')}
            f={k:np.asarray(fr[k][i],dtype='float64') for k in ('pan','lms','ms')}
            bands=8 if case['sensor'] in ('WV3','WV2') else 4
            for a in (r,f):
                c,h,w=a['lms'].shape
                if c!=bands or a['pan'].shape!=(1,h,w) or a['ms'].shape!=(c,h//4,w//4) or not all(np.isfinite(v).all() for v in a.values()):
                    raise ValueError('Invalid NCHW scene')
            y=predict(r);store('RR',i,r,y,rr_metrics(r['gt'],y))
            pan,lms,ms=wald.degrade(r['pan'],r['ms'],'cpu')
            a=dict(pan=pan,lms=lms,ms=ms,gt=r['ms']);y=predict(a)
            store('RR2',i,a,y,rr_metrics(a['gt'],y))
            pan,lms,ms=wald.degrade(f['pan'],f['ms'],'cpu')
            a=dict(pan=pan,lms=lms,ms=ms,gt=f['ms']);y=predict(a)
            store('FR_WALD',i,a,y,rr_metrics(a['gt'],y))
            y=predict(f)
            store('FR',i,f,y,fr_metrics(y,f['lms'],f['ms'],f['pan']),
                  dict(FR_OBS_RMSE_DN=wald.observation_rmse(y,f['ms'],'cpu',5)))
            elapsed=time.monotonic()-started
            eta=elapsed/(i+1)*((limit or 20)-i-1)
            print(f"{case['id']} scene {i+1}/{limit or 20}: {time.monotonic()-t:.1f}s; inference remaining ~{eta/60:.1f} min; MATLAB/setup not included",flush=True)
    imported={}
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if path:
            p=Path(path).resolve()
            if p.is_relative_to(REPO) and p.is_file() and p.suffix=='.py':imported[p.relative_to(REPO).as_posix()]=sha(p)
    # importlib-loaded retained modules need not be registered in sys.modules.
    explicit=[HERE/'suite.py',HERE/'report.py',HERE/'score_suite.m',EXP[53]/'model.py',EXP[54]/'model.py',
              EXP[54]/'prepare.py',EXP[43]/'model.py',EXP[48]/'model.py',
              PERF/'29_capacity_refinement/calibration.json',SSA/'research/performance/28_residual_refinement/model.py']
    for path in explicit:imported[path.relative_to(REPO).as_posix()]=sha(path)
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    meta=dict(case=case,checkpoint=cp.relative_to(REPO).as_posix(),checkpoint_sha256=expected,
        checkpoint_best_epoch=ck.get('best_epoch',ck.get('epoch')),data=data_files(case['sensor']),
        files=files,source_sha256=imported,source_commit=git_head,peak=peak,bits=bits,ratio=4,block=32,
        scenes=limit or 20,inference='CPU FP32 full scene; validation-selected retained best; no clipping, TTA, or training',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),
        inference_seconds=time.monotonic()-started,dlpan_commit=DLPAN_REV,
        legacy_psnr='per-band mean, peak 2047 even for GF2',
        matlab_psnr='both global-MSE and band-mean at sensor peak; definitions kept separate')
    write(output/'prediction_manifest.json',meta)
    write(output/'python_legacy.json',records)
    # Compact previous numeric report, not the checkpoint's giant training history.
    write(output/'previous_report.json',old_rows(case))
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    m=sub.add_parser('matrix');m.add_argument('--experiments',default='53,54');m.add_argument('--output',type=Path)
    e=sub.add_parser('export');e.add_argument('--case',required=True);e.add_argument('--out',type=Path,required=True)
    e.add_argument('--data',type=Path,required=True);e.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.mode=='matrix':
        obj=dict(include=cases(tuple(int(x) for x in args.experiments.split(','))))
        if args.output:write(args.output,obj)
        else:print(json.dumps(obj,separators=(',',':')))
    else:
        case=next((x for x in cases() if x['id']==args.case),None)
        if case is None:raise ValueError('Unknown retained experiment case')
        export(case,args.out,args.data,args.threads)


if __name__=='__main__':main()
