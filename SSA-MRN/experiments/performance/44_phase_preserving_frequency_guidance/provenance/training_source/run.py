"""Windows controller: immutable source/data contract, shared mmap cache, two slots."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST
from datetime import datetime,timedelta

def source_hashes():
    paths=[HERE/'model.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',*sorted((HERE/'provenance/source').rglob('*.py'))]
    return {str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in paths}
def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def prepare():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True);cache=root/'cache';cache.mkdir(exist_ok=True)
    if (root/'runtime_config.json').exists() and any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('run already initialized; prepare cannot change it')
    records={}
    for split,name in cfg['data'].items():
        p=Path(cfg['data_root'])/name;print('HASH '+split,flush=True);digest=sha(p)
        if digest!=cfg['data_sha256'][split]:raise ValueError('data sha mismatch '+split)
        with h5py.File(p) as f:
            expected={'train':17139,'val':1905,'RR':20,'FR':20}[split]
            if len(f['pan'])!=expected:raise ValueError('count '+split)
            records[split]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,arrays={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f})
            if split in ('train','val'):
                for k in KEYS:
                    dest=cache/f'{split}_{k}.npy';temp=cache/f'{split}_{k}_building.npy';ds=f[k]
                    arr=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=ds.shape)
                    for i in range(0,len(ds),128):
                        x=np.asarray(ds[i:i+128],dtype=np.float32)/2047.
                        if not np.isfinite(x).all():raise FloatingPointError('data nonfinite')
                        arr[i:i+len(x)]=x
                    arr.flush();del arr;os.replace(temp,dest)
                    records[split]['arrays'][k]['normalized_cache_sha256']=sha(dest)
                    print('CACHE '+split+' '+k,flush=True)
    cfg['layout']='contiguous';cfg['cache_manifest_sha256']=None;cfg['source_hashes']=source_hashes();cfg['base_main']=json.loads((HERE/'provenance/base.json').read_text())['main_sha'];cfg['created_kst']=now()
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    import torch
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cache='read-only shared float32 NPY mmap; CPU pinned per-batch transfer, no per-worker full dataset copy'))
    print('PREPARE PASS',flush=True)
def optimize():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after initialization')
    results={}
    for layout,threads in [('contiguous',1),('channels_last',1),('channels_last',2)]:
        jobs=[];start=time.monotonic();key=f'{layout}_t{threads}'
        for arm in ('F1','F2'):
            log=(out/f'benchmark_{key}_{arm}.log').open('w',encoding='utf8')
            env={**os.environ,'CTRS_THREADS':str(threads),'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads)}
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--arm',arm,'--layout',layout],stdout=log,stderr=subprocess.STDOUT,env=env);jobs.append((p,log))
        codes=[]
        for p,log in jobs:codes.append(p.wait());log.close()
        rows=[]
        for arm in ('F1','F2'):
            p=out/f'benchmark_{key}_{arm}.json'
            if p.exists():rows.append(json.loads(p.read_text(encoding='utf8')))
        safe=all(x==0 for x in codes) and len(rows)==2 and sum(x['peak_reserved_bytes'] for x in rows)<6*2**30
        results[key]=dict(layout=layout,threads=threads,returncodes=codes,wall_seconds=time.monotonic()-start,rows=rows,safe=safe,score=max(x['median_step_seconds'] for x in rows) if safe else None)
        print(json.dumps({key:results[key]}),flush=True)
    safe=[k for k,v in results.items() if v['safe']]
    if not safe:raise RuntimeError('two-process benchmark failed; no training started')
    chosen=min(safe,key=lambda x:results[x]['score']);cfg['layout']=results[chosen]['layout'];cfg['cpu_threads']=results[chosen]['threads'];cfg['optimization']=dict(selected=chosen,results=results,note='two concurrent frequency arms, fixed FP32 batch32; throughput only')
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    os.environ['CTRS_THREADS']=str(cfg['cpu_threads'])
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),selected=chosen,results=results,fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS '+chosen,flush=True)

def snapshot(cfg,active,phase):
    root=Path(cfg['runtime']);rows=[];durations={}
    for arm,seed in cfg['queue']:
        out=root/'runs'/f'{arm}_s{seed}'
        h=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
        ds=[x['epoch_seconds'] for x in h if 'epoch_seconds' in x]
        if ds:durations[arm]=float(np.median(ds[-3:]))
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        rows.append(dict(arm=arm,seed=seed,training_complete=(out/'training_complete.json').exists(),evaluation_complete=(out/'evaluation_complete.json').exists(),history_epochs=len([x for x in h if x['epoch']>0]),status=status))
    loads=[0.,0.];pending=[]
    for row in rows:
        if row['training_complete']:continue
        sec=durations.get(row['arm'],max(durations.values()) if durations else None)
        if sec is None:pending=None;break
        remain=sec*(cfg['epochs']-row['history_epochs'])
        if (row['arm'],row['seed']) in active:loads[loads.index(min(loads))]+=remain
        else:pending.append(remain)
    remaining=None
    if pending is not None:
        for sec in pending:loads[loads.index(min(loads))]+=sec
        remaining=max(loads)
    write(root/'queue_status.json',dict(time_kst=now(),phase=phase,active=[dict(arm=a,seed=s) for a,s in active],runs=rows,epoch_seconds_by_arm=durations,training_remaining_seconds=remaining,training_end_kst=(datetime.now(KST)+timedelta(seconds=remaining)).isoformat() if remaining is not None else None,eta_basis='recent complete epochs; slowest observed arm proxy for pending; excludes final evaluation'))

def run(resume=False):
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    os.environ['CTRS_THREADS']=str(cfg['cpu_threads']);os.environ['OMP_NUM_THREADS']=str(cfg['cpu_threads']);os.environ['MKL_NUM_THREADS']=str(cfg['cpu_threads'])
    active={};pending=[]
    for arm,seed in cfg['queue']:
        out=root/'runs'/f'{arm}_s{seed}';done=out/'training_complete.json'
        if done.exists():
            if json.loads(done.read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs')
        else:pending.append((arm,seed))
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],max_parallel=2,resume=resume))
    def drain(phase,jobs):
        pending=list(jobs);running={}
        while pending or running:
            while pending and len(running)<2:
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
    # Test is not used to alter the precommitted twelve-run queue or checkpoint selection.
    drain('evaluate',[(a,s) for a,s in cfg['queue'] if not(root/'runs'/f'{a}_s{s}'/'evaluation_complete.json').exists()])
    snapshot(cfg,{},'complete');write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],training_runs=len(cfg['queue']),RR_scenes=20*len(cfg['queue']),FR_scenes=20*len(cfg['queue'])))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','optimize','run'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='optimize':optimize()
    else:run(a.resume)
