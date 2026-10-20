"""Windows controller (from 48): immutable source/data contract, 43-identical cache plus a coarse (x2) cache."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST
from datetime import datetime,timedelta

def source_hashes():
    perf=HERE.parent;e43=perf/'43_normalized_parallel_capacity'
    root=perf.parents[1];e48=perf/'48_parallel_capacity_observation_refinement'
    paths=[HERE/'model.py',HERE/'coarse.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',e48/'model.py',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),root/'research/performance/28_residual_refinement/model.py',perf/'29_capacity_refinement/calibration.json',root/'common/observation.py',root/'common/interpolation.py']
    return {str(p.relative_to(perf.parents[1])).replace('\\','/'):sha(p) for p in paths}
def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def prepare():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True);cache=root/'cache';cache.mkdir(exist_ok=True)
    if (root/'runtime_config.json').exists() and any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('run already initialized; prepare cannot change it')
    # Rebuild the 43 normalized cache with the 43 procedure and require byte-identical arrays (43 manifest hashes).
    source_path=HERE.parents[2]/cfg['cache_manifest_source']
    if sha(source_path)!=cfg['cache_manifest_source_sha256']:raise ValueError('43 cache manifest changed')
    source=json.loads(source_path.read_text(encoding='utf8'));records={}
    for split,name in cfg['data'].items():
        p=Path(cfg['data_root'])/name;print('HASH '+split,flush=True);digest=sha(p)
        if digest!=cfg['data_sha256'][split] or source[split]['sha256']!=digest:raise ValueError('data sha mismatch '+split)
        with h5py.File(p) as f:
            expected={'train':17139,'val':1905,'RR':20,'FR':20}[split]
            if len(f['pan'])!=expected:raise ValueError('count '+split)
            records[split]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,arrays={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f})
            if split in ('train','val'):
                for k in KEYS:
                    dest=cache/f'{split}_{k}.npy';want=source[split]['arrays'][k]['normalized_cache_sha256']
                    if not (dest.exists() and sha(dest)==want):
                        temp=cache/f'{split}_{k}_building.npy';ds=f[k]
                        arr=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=ds.shape)
                        for i in range(0,len(ds),128):
                            x=np.asarray(ds[i:i+128],dtype=np.float32)/2047.
                            if not np.isfinite(x).all():raise FloatingPointError('data nonfinite')
                            arr[i:i+len(x)]=x
                        arr.flush();del arr;os.replace(temp,dest)
                    got=sha(dest)
                    if got!=want:raise ValueError('cache differs from 43 '+split+' '+k)
                    records[split]['arrays'][k]['normalized_cache_sha256']=got
                    print('CACHE '+split+' '+k+' equal to 43',flush=True)
                import torch
                from coarse import CoarsePairs
                cp=CoarsePairs('cuda');src={k:np.load(cache/f'{split}_{k}.npy',mmap_mode='r') for k in KEYS};n=len(src['pan'])
                shapes={'pan':(n,1,32,32),'lms':(n,4,32,32),'ms':(n,4,8,8),'gt':(n,4,32,32)};temp={k:cache/f'coarse_{split}_{k}_building.npy' for k in KEYS}
                arrs={k:np.lib.format.open_memmap(temp[k],mode='w+',dtype=np.float32,shape=shapes[k]) for k in KEYS}
                for i in range(0,n,256):
                    pan2,lms2,ms2,gt2=cp(torch.from_numpy(np.array(src['pan'][i:i+256])).cuda(),torch.from_numpy(np.array(src['gt'][i:i+256])).cuda())
                    for k,v in zip(('pan','lms','ms','gt'),(pan2,lms2,ms2,gt2)):
                        v=v.cpu().numpy()
                        if not np.isfinite(v).all():raise FloatingPointError('coarse nonfinite')
                        arrs[k][i:i+len(v)]=v
                records[split]['coarse']={}
                for k in KEYS:
                    arrs[k].flush();del arrs[k];os.replace(temp[k],cache/f'coarse_{split}_{k}.npy')
                    records[split]['coarse'][k]=dict(shape=list(shapes[k]),sha256=sha(cache/f'coarse_{split}_{k}.npy'))
                    print('COARSE '+split+' '+k,flush=True)
    cfg['source_hashes']=source_hashes();cfg['created_kst']=now()
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    import torch
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cache='rebuilt float32 NPY mmap, byte-identical to the 43 cache manifest'))
    print('PREPARE PASS',flush=True)
def optimize():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    # Concurrency only: batch, AMP, order, determinism, validation and layout stay at the 43 values, so results are unaffected.
    results={};layout=cfg['layout']
    for count in cfg['concurrency_candidates']:
        arms=[cfg['queue'][i%len(cfg['queue'])][0] for i in range(count)];tag=f'n{count}';jobs=[];start=time.monotonic()
        for k,arm in enumerate(arms):
            log=(out/f'benchmark_{tag}_{k}_{arm}.log').open('w',encoding='utf8')
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--arm',arm,'--layout',layout,'--tag',f'{tag}_{k}'],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
        codes=[]
        for p,log in jobs:codes.append(p.wait());log.close()
        rows=[json.loads((out/f'benchmark_{tag}_{k}_{arm}.json').read_text(encoding='utf8')) for k,arm in enumerate(arms) if (out/f'benchmark_{tag}_{k}_{arm}.json').exists()]
        reserved=sum(x['peak_reserved_bytes'] for x in rows)
        safe=all(x==0 for x in codes) and len(rows)==count and reserved<cfg['vram_budget_gib']*2**30
        # Aggregate patches per second weighted equally per job; per-arm speeds give the queue-level estimate below.
        results[count]=dict(arms=arms,returncodes=codes,wall_seconds=time.monotonic()-start,rows=rows,safe=safe,reserved_gib=reserved/2**30,patches_per_second=sum(cfg['batch']/x['median_step_seconds'] for x in rows) if safe else None)
        print(json.dumps({count:{k:v for k,v in results[count].items() if k!='rows'}}),flush=True)
    single={x['arm']:x['median_step_seconds'] for x in results.get(1,{}).get('rows',[])}
    def finish(count):
        # Queue makespan proxy: each job needs the same number of steps; step time scales with the measured slowdown at this concurrency.
        rows=results[count]['rows'];slow={};[slow.setdefault(x['arm'],[]).append(x['median_step_seconds']) for x in rows]
        per={a:float(np.median(v)) for a,v in slow.items()};pending=[a for a,s in cfg['queue']];loads=[0.]*count
        for a in pending:
            t=per.get(a,max(per.values()));i=min(range(count),key=lambda x:loads[x]);loads[i]+=t
        return max(loads)
    safe=[c for c,v in results.items() if v['safe']]
    if not safe:raise RuntimeError('concurrency benchmark failed, no training started')
    scores={c:finish(c) for c in safe};best=min(scores.values());chosen=min(c for c in safe if scores[c]<=best*1.03)
    cfg['max_parallel']=chosen;cfg['optimization']=dict(selected_max_parallel=chosen,makespan_proxy_step_units=scores,results={str(k):v for k,v in results.items()},note='concurrency sweep with the real data path; smallest count within 3% of the fastest queue makespan proxy; no validation/test tuning')
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),selected_max_parallel=chosen,makespan_proxy_step_units=scores,results={str(k):v for k,v in results.items()},fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS max_parallel='+str(chosen),flush=True)
def snapshot(cfg,active,phase):
    root=Path(cfg['runtime']);rows=[];known={};remaining=[]
    for arm,seed in cfg['queue']:
        out=root/'runs'/f'{arm}_s{seed}';history=out/'history.json';status=out/'status.json';row=dict(arm=arm,seed=seed,training_complete=(out/'training_complete.json').exists(),evaluation_complete=(out/'evaluation_complete.json').exists())
        h=json.loads(history.read_text(encoding='utf8')) if history.exists() else []
        durations=[x['epoch_seconds'] for x in h if 'epoch_seconds' in x]
        if durations:known.setdefault(arm,[]).extend(durations[-3:])
        if status.exists():
            try:row['status']=json.loads(status.read_text(encoding='utf8'))
            except (json.JSONDecodeError,PermissionError):pass
        rows.append(row)
    estimates={}
    for arm in ('C2MS','F1MS'):
        samples=known.get(arm,known.get('F1MS' if arm=='C2MS' else 'C2MS',[]))
        estimates[arm]=float(np.median(samples)) if samples else None
    loads=[];pending=[]
    for row in rows:
        if row['training_complete']:continue
        st=row.get('status',{});arm=row['arm'];speed=estimates[arm]
        if speed is None:remaining=[];break
        epoch=st.get('epoch',0);batch=st.get('batch',0) if st.get('phase')=='train' else 0;steps=st.get('batches',536)
        completed=max(0,epoch-1+batch/steps) if st.get('phase')=='train' else epoch if st.get('phase')=='epoch_complete' else 0
        sec=max(0,cfg['epochs']-completed)*speed
        if (arm,row['seed']) in active:loads.append(sec)
        else:pending.append(sec)
    else:
        while len(loads)<cfg['max_parallel']:loads.append(0.)
        for sec in pending:
            i=min(range(len(loads)),key=lambda x:loads[x]);loads[i]+=sec
        remaining=max(loads)
    eta=datetime.now(KST)+timedelta(seconds=remaining) if isinstance(remaining,(int,float)) else None
    result=dict(time_kst=now(),phase=phase,active=[dict(arm=a,seed=s) for a,s in active],runs=rows,epoch_seconds_by_arm=estimates,training_remaining_seconds=remaining if isinstance(remaining,(int,float)) else None,training_end_kst=eta.isoformat() if eta else None,eta_basis='recent full train+validation+checkpoint epochs, capacity-matched speed proxy for not-yet-started arms; excludes final RR/FR evaluation')
    write(root/'queue_status.json',result);return result

def run(resume=False):
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    active={};pending=[]
    for arm,seed in cfg['queue']:
        out=root/'runs'/f'{arm}_s{seed}';done=out/'training_complete.json'
        if done.exists():
            if json.loads(done.read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs')
        else:pending.append((arm,seed))
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],max_parallel=cfg['max_parallel'],resume=resume))
    def drain(phase,jobs):
        pending=list(jobs);running={}
        while pending or running:
            while pending and len(running)<cfg['max_parallel']:
                arm,seed=pending.pop(0);out=root/'runs'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True);log=(out/(phase+'.log')).open('a',encoding='utf8')
                args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',phase,'--arm',arm,'--seed',str(seed)]
                if phase=='train' and resume:args+=['--resume']
                p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);running[(arm,seed)]=(p,log)
                print(json.dumps(dict(time_kst=now(),phase='launched_'+phase,arm=arm,seed=seed,pid=p.pid)),flush=True)
            for key,(p,log) in list(running.items()):
                code=p.poll()
                if code is not None:
                    log.close();del running[key]
                    if code!=0:
                        write(root/'controller_failure.json',dict(time_kst=now(),phase=phase,job=key,returncode=code,remaining_jobs=pending,active_jobs=list(running),note='other active job is preserved; no new jobs launched'))
                        raise RuntimeError('job failed; other active job preserved: '+str(key))
            snapshot(cfg,running,phase);time.sleep(3)
    drain('train',pending)
    # Test is not used to alter the precommitted queue or checkpoint selection.
    drain('evaluate',[(a,s) for a,s in cfg['queue'] if not(root/'runs'/f'{a}_s{s}'/'evaluation_complete.json').exists()])
    snapshot(cfg,{},'complete');write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],training_runs=len(cfg['queue']),RR_scenes=20*len(cfg['queue']),FR_scenes=20*len(cfg['queue'])))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','optimize','run'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='optimize':optimize()
    else:run(a.resume)
