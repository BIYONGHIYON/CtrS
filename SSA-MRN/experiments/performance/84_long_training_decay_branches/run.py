"""Windows controller (from 83): QB, GF2 and WV3 caches rebuilt from the H5 and checked against the 63 (QB) and 65 stage 3 (GF2, WV3) hashes,
sealed config, then two GPU lanes: lane 1 trains the TRUNK runs one sensor after another (QB, GF2, WV3); lane 2 trains each branch as soon as
its trunk snapshot exists and evaluates every finished run (branches first, in sensor order). Nothing is tuned while it runs; summarize()
picks the training length per sensor by validation only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST,run_name,epochs,branch_start,snapshot_dir
from prepare import SENSORS,data_files
from model import ARMS,BRANCHES
from datetime import datetime,timedelta

def source_hashes():
    perf=HERE.parent;root=perf.parents[1];e43=perf/'43_normalized_parallel_capacity';e54=perf/'54_multi_sensor_scale_mixed_training'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',perf/'83_factorized_c2ms_finetuning/prepare.py',
           perf/'79_deep_narrow_branch_projection/prepare.py',perf/'82_cross_sensor_band_calibration/prepare.py',perf/'58_spectral_loss_multi_sensor/train.py',
           e54/'model.py',e54/'prepare.py',e54/'train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),
           root/'common/interpolation.py',root/'common/metrics.py',root/'common/observation.py',root/'common/evaluation_protocol.py']
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    paths += [root/p for p in conf['cache_reference_manifest'].values()]+[perf/p for p in conf['calibration_reference'].values()]
    paths += [root/conf['control'][sensor].replace('<seed>',str(s)) for sensor in conf['sensors'] for s in conf['seeds']]
    paths = list(dict.fromkeys(paths))
    return {str(p.relative_to(root)).replace('\\','/'):sha(p) for p in paths}
def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def prepare():
    # Rebuild the caches from the H5 as experiments 73-79 did, then compare every
    # array with the cache hashes recorded by experiment 65 stage 3 (GF2, WV3). Normalized arrays must match; coarse arrays are recorded (GPU/torch dependent).
    import torch,shutil
    from prepare import CoarsePairs
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True)
    if (root/'runtime_config.json').exists():raise RuntimeError('runtime already prepared; do not overwrite')
    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('less than 20 GiB free')
    old={s:json.loads((HERE.parents[2]/p).read_text(encoding='utf8'))[s] for s,p in cfg['cache_reference_manifest'].items()};records={};matches={}
    for target in cfg['transfer_tests']:
        for split in ('RR','FR'):
            p=Path(cfg['data_root'])/data_files(target)[split];digest=sha(p);print('HASH',target,split,flush=True)
            if digest!=cfg['transfer_data_sha256'][target][split]:raise ValueError('transfer data sha mismatch '+target+' '+split)
    for sensor in cfg['sensors']:
        cache=root/'cache'/sensor;cache.mkdir(parents=True,exist_ok=True);maxv=SENSORS[sensor]['max_value'];records[sensor]={}
        for split,name in data_files(sensor).items():
            p=Path(cfg['data_root'])/name;print('HASH',sensor,split,flush=True);digest=sha(p)
            if digest!=cfg['data_sha256'][sensor][split] or digest!=old[sensor][split]['sha256']:raise ValueError('data sha mismatch '+sensor+' '+split)
            with h5py.File(p) as f:
                if len(f['pan'])!=cfg['data_counts'][sensor][split]:raise ValueError('count '+sensor+' '+split)
                rec=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,arrays={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f})
                if split in ('train','val'):
                    for k in KEYS:
                        dest=cache/f'{split}_{k}.npy';temp=cache/f'{split}_{k}_building.npy';ds=f[k]
                        arr=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=ds.shape)
                        for i in range(0,len(ds),128):
                            x=np.asarray(ds[i:i+128],dtype=np.float32)/np.float32(maxv)
                            if not np.isfinite(x).all():raise FloatingPointError('data nonfinite')
                            arr[i:i+len(x)]=x
                        arr.flush();del arr;os.replace(temp,dest);h=sha(dest);rec['arrays'][k]['normalized_cache_sha256']=h
                        if h!=old[sensor][split]['arrays'][k]['normalized_cache_sha256']:raise ValueError('normalized cache differs from the reference manifest '+str(dest))
                        print('CACHE',sensor,split,k,flush=True)
                    cp=CoarsePairs(sensor,'cuda');src={k:np.load(cache/f'{split}_{k}.npy',mmap_mode='r') for k in KEYS};n=len(src['pan']);nb=SENSORS[sensor]['bands']
                    shapes={'pan':(n,1,32,32),'lms':(n,nb,32,32),'ms':(n,nb,8,8),'gt':(n,nb,32,32)};temp={k:cache/f'coarse_{split}_{k}_building.npy' for k in KEYS}
                    arrs={k:np.lib.format.open_memmap(temp[k],mode='w+',dtype=np.float32,shape=shapes[k]) for k in KEYS}
                    for i in range(0,n,256):
                        out=cp(torch.from_numpy(np.array(src['pan'][i:i+256])).cuda(),torch.from_numpy(np.array(src['gt'][i:i+256])).cuda())
                        for k,v in zip(('pan','lms','ms','gt'),out):
                            v=v.cpu().numpy()
                            if not np.isfinite(v).all():raise FloatingPointError('coarse nonfinite')
                            arrs[k][i:i+len(v)]=v
                    rec['coarse']={}
                    for k in KEYS:
                        arrs[k].flush();del arrs[k];os.replace(temp[k],cache/f'coarse_{split}_{k}.npy');h=sha(cache/f'coarse_{split}_{k}.npy')
                        rec['coarse'][k]=dict(shape=list(shapes[k]),sha256=h);matches[f'{sensor}_{split}_coarse_{k}']=h==old[sensor][split]['coarse'][k]['sha256'];print('COARSE',sensor,split,k,flush=True)
            records[sensor][split]=rec
    from common.evaluation_protocol import OBS_MARGIN
    from prepare import OBS_MARGIN as P_MARGIN
    if OBS_MARGIN!=P_MARGIN:raise AssertionError('projection margin differs from common/evaluation_protocol')
    for sensor in cfg['sensors']:
        for seed in cfg['seeds']:
            p=HERE.parents[2]/cfg['control'][sensor].replace('<seed>',str(seed))
            if sha(p)!=cfg['control_sha256'][f'{sensor}_s{seed}']:raise ValueError('control weights differ '+str(p))
    cfg['source_hashes']=source_hashes();cfg['created_kst']=now();cfg['coarse_cache_equal_to_reference']=matches
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cache='rebuilt from H5; normalized arrays equal the 63 (QB) and 65 stage 3 (GF2, WV3) hashes; coarse equality recorded',free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS',json.dumps(matches),flush=True)

def optimize():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    # Single-job step benchmark per sensor (every run is the same C2MS network); used only for the ETA.
    rows={}
    for sensor in cfg['sensors']:
        log=(out/f'benchmark_{sensor}.log').open('w',encoding='utf8')
        code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--sensor',sensor,'--arm','TRUNK','--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
        if code:raise RuntimeError('benchmark failed '+sensor)
        rows[sensor]=json.loads((out/f'benchmark_n1_{sensor}_TRUNK.json').read_text(encoding='utf8'))
    cfg['benchmark']={k:dict(median_step_seconds=v['median_step_seconds'],steps_per_epoch=v['steps_per_epoch'],peak_reserved_gib=v['peak_reserved_bytes']/2**30) for k,v in rows.items()}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),benchmark=cfg['benchmark'],fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS',json.dumps(cfg['benchmark']),flush=True)

def runs(cfg):return [(s,a,x) for x in cfg['seeds'] for s in cfg['sensors'] for a in ARMS]
def remaining_epochs(cfg,root,sensor,arm,seed):
    out=root/'runs'/run_name(sensor,arm,seed)
    if (out/'training_complete.json').exists():return 0.
    total=epochs(cfg,arm);start=0 if arm=='TRUNK' else branch_start(cfg,arm)[0];done=start
    try:
        st=json.loads((out/'status.json').read_text(encoding='utf8'))
        if st.get('phase')=='train':done=st['epoch']-1+st.get('batch',0)/max(st.get('batches',1),1)
        elif st.get('phase')=='epoch_complete':done=st['epoch']
    except (FileNotFoundError,json.JSONDecodeError,PermissionError,KeyError):pass
    return max(0.,total-done)
def snapshot(cfg,active,phase):
    root=Path(cfg['runtime']);rows=[];known={}
    for sensor,arm,seed in runs(cfg):
        out=root/'runs'/run_name(sensor,arm,seed);row=dict(sensor=sensor,arm=arm,seed=seed,training_complete=(out/'training_complete.json').exists(),evaluation_complete=(out/'evaluation_complete.json').exists(),remaining_epochs=remaining_epochs(cfg,root,sensor,arm,seed))
        try:
            h=json.loads((out/'history.json').read_text(encoding='utf8'));d=[x['epoch_seconds'] for x in h if 'epoch_seconds' in x]
            if d:known.setdefault(sensor,[]).extend(d[-3:])
        except (FileNotFoundError,json.JSONDecodeError,PermissionError):pass
        rows.append(row)
    # Epoch seconds while two jobs share the GPU: measured recent epochs of the sensor, else the single-job benchmark x 1.15 x 1.7 (74/75).
    est={s:(float(np.median(known[s])) if s in known else v['median_step_seconds']*v['steps_per_epoch']*1.15*1.7) for s,v in cfg['benchmark'].items()}
    trunk=sum(r['remaining_epochs']*est[r['sensor']] for r in rows if r['arm']=='TRUNK');other=sum(r['remaining_epochs']*est[r['sensor']] for r in rows if r['arm']!='TRUNK')
    remaining=max(trunk+min(other,est[cfg['sensors'][-1]]*epochs(cfg,'B1000')*(1-cfg['branch_fraction'])),(trunk+other)/2)
    result=dict(time_kst=now(),phase=phase,active=[dict(sensor=s,arm=a,seed=x,job=j) for s,a,x,j in active],runs=rows,epoch_seconds=est,
                trunk_lane_remaining_seconds=trunk,branch_lane_remaining_seconds=other,training_end_kst=(datetime.now(KST)+timedelta(seconds=remaining)).isoformat(),
                eta_basis='rough: lane 1 trunk epochs, lane 2 branch epochs, measured epoch seconds per sensor while two jobs run; excludes evaluations')
    write(root/'queue_status.json',result);return result

def run(resume=False):
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],resume=resume))
    def done(sensor,arm,seed,kind):return (root/'runs'/run_name(sensor,arm,seed)/f'{kind}_complete.json').exists()
    def launch(phase,sensor,arm,seed):
        out=root/'runs'/run_name(sensor,arm,seed);out.mkdir(parents=True,exist_ok=True);log=(out/(phase+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',phase,'--sensor',sensor,'--arm',arm,'--seed',str(seed)]
        if phase=='train' and arm=='TRUNK' and resume and (out/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);print(json.dumps(dict(time_kst=now(),phase='launched_'+phase,sensor=sensor,arm=arm,seed=seed,pid=p.pid)),flush=True)
        return p,log
    for sensor,arm,seed in runs(cfg):
        f=root/'runs'/run_name(sensor,arm,seed)/'training_complete.json'
        if f.exists() and json.loads(f.read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs')
    def next_trunk():
        for sensor,arm,seed in runs(cfg):
            if arm=='TRUNK' and not done(sensor,arm,seed,'training'):return ('train',sensor,arm,seed)
    def next_other(busy):
        """Branch evaluations and trainings first (sensor order, shortest first), then trunk evaluations."""
        for sensor,arm,seed in runs(cfg):
            if arm=='TRUNK' or (sensor,arm,seed) in busy:continue
            if done(sensor,arm,seed,'training'):
                if not done(sensor,arm,seed,'evaluation'):return ('evaluate',sensor,arm,seed)
            elif (snapshot_dir(cfg,sensor,seed,branch_start(cfg,arm)[0])/'snapshot.json').exists():return ('train',sensor,arm,seed)
        for sensor,arm,seed in runs(cfg):
            if arm=='TRUNK' and (sensor,arm,seed) not in busy and done(sensor,arm,seed,'training') and not done(sensor,arm,seed,'evaluation'):return ('evaluate',sensor,arm,seed)
    lanes={'trunk':None,'other':None}
    while True:
        busy={tuple(v[0][1:]) for v in lanes.values() if v}
        if lanes['trunk'] is None:
            job=next_trunk()
            if job:lanes['trunk']=(job,*launch(*job))
        if lanes['other'] is None:
            job=next_other(busy)
            if job:lanes['other']=(job,*launch(*job))
        if not any(lanes.values()):break
        snapshot(cfg,[(*v[0][1:],v[0][0]) for v in lanes.values() if v],'run');time.sleep(10)
        for lane,v in list(lanes.items()):
            if v is None or v[1].poll() is None:continue
            (phase,sensor,arm,seed),p,log=v;log.close();lanes[lane]=None
            if p.returncode or not done(sensor,arm,seed,'training' if phase=='train' else 'evaluation'):
                for w in lanes.values():
                    if w:w[1].wait();w[2].close()
                write(root/'controller_failure.json',dict(time_kst=now(),phase=phase,job=[sensor,arm,seed],returncode=p.returncode));raise RuntimeError(f'{phase} failed {sensor} {arm} {seed}')
    missing=[r for r in runs(cfg) if not done(*r,'evaluation')]
    if missing:raise RuntimeError('queue ended with unfinished runs: '+json.dumps(missing))
    from evaluate import summarize
    summarize(cfg);snapshot(cfg,[],'complete');write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(runs(cfg))))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','optimize','run','summarize'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='optimize':optimize()
    elif a.mode=='summarize':
        from evaluate import summarize
        cfg=json.loads((Path(json.loads((HERE/'config.json').read_text(encoding='utf8'))['runtime'])/'runtime_config.json').read_text(encoding='utf8'));print(json.dumps(summarize(cfg),ensure_ascii=False)[:2000])
    else:run(a.resume)
