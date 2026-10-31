"""Sealed sequential controller: all training, validation screening, then test reporting."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST,run_name,epochs
from prepare import SENSORS,data_files
from model import ARMS
from datetime import datetime,timedelta

def source_hashes():
    perf=HERE.parent;root=perf.parents[1];e43=perf/'43_normalized_parallel_capacity';e54=perf/'54_multi_sensor_scale_mixed_training'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',HERE/'validation.py',perf/'70_c2ms_dilated_backbone/model.py',e54/'model.py',e54/'prepare.py',e54/'train.py',perf/'58_spectral_loss_multi_sensor/train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),root/'common/interpolation.py',root/'common/metrics.py',root/'common/observation.py',root/'common/evaluation_protocol.py',root/'common/result_images.py']
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    templates=[*conf['controls'].values(),*(q for refs in conf.get('reference_controls',{}).values() for q in refs.values())]
    paths += [root/p.replace('<seed>',str(s)) for p in templates for s in (conf['seeds'] if '<seed>' in p else [None])]
    return {str(p.relative_to(root)).replace('\\','/'):sha(p) for p in paths}
def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def prepare():
    # Use independently named hard links after hash checks: no future reference to the previous runtime.
    import torch,shutil
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True)
    if (root/'runtime_config.json').exists():raise RuntimeError('runtime already prepared; do not overwrite')
    source=Path(cfg['cache_reuse_source']);old=json.loads((source/'data_manifest.json').read_text(encoding='utf8'))
    records={}
    if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('less than 5 GiB free')
    for sensor in cfg['sensors']:
        records[sensor]={};cache=root/'cache'/sensor;cache.mkdir(parents=True,exist_ok=True)
        for split,name in data_files(sensor).items():
            path=Path(cfg['data_root'])/name;digest=sha(path)
            if digest!=cfg['data_sha256'][sensor][split] or digest!=old[sensor][split]['sha256']:raise ValueError('data sha mismatch '+sensor+split)
            with h5py.File(path) as f:
                if len(f['pan'])!=cfg['data_counts'][sensor][split]:raise ValueError('sample count')
            rec=json.loads(json.dumps(old[sensor][split]));rec['path']=str(path);records[sensor][split]=rec
            if split in ('train','val'):
                for coarse in (False,True):
                    for k in KEYS:
                        name=f"{'coarse_' if coarse else ''}{split}_{k}.npy";src=source/'cache'/sensor/name;dest=cache/name
                        expected=rec['coarse'][k]['sha256'] if coarse else rec['arrays'][k]['normalized_cache_sha256']
                        if sha(src)!=expected:raise ValueError('source cache mismatch '+str(src))
                        if not dest.exists():os.link(src,dest)
                        if sha(dest)!=expected:raise ValueError('cache mismatch '+str(dest))
                        arr=np.load(dest,mmap_mode='r')
                        shape=rec['coarse'][k]['shape'] if coarse else rec['arrays'][k]['shape']
                        if list(arr.shape)!=shape or arr.dtype!=np.float32:raise ValueError('cache shape/dtype')
                print('VERIFIED CACHE',sensor,split,flush=True)
    # Transfer test files are verified now as well, without being used by training/selection.
    for sensor,table in cfg['transfer_data_sha256'].items():
        for split,digest in table.items():
            if sha(Path(cfg['data_root'])/data_files(sensor)[split])!=digest:raise ValueError('transfer data sha mismatch')
    cfg['source_hashes']=source_hashes();cfg['created_kst']=now()
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cache='independent hard links; all data/cache SHA256 verified; new runtime paths',free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS',flush=True)

def optimize():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    # Single-job step benchmark per sensor and arm (the GPU is saturated by one job); used only for the ETA.
    rows={}
    for sensor in cfg['sensors']:
        for arm in ARMS:
            log=(out/f'benchmark_{sensor}_{arm}.log').open('w',encoding='utf8')
            code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--sensor',sensor,'--arm',arm,'--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
            if code:raise RuntimeError('benchmark failed '+sensor+' '+arm)
            rows[f'{sensor}_{arm}']=json.loads((out/f'benchmark_n1_{sensor}_{arm}.json').read_text(encoding='utf8'))
    cfg['max_parallel']=1;cfg['benchmark']={k:dict(median_step_seconds=v['median_step_seconds'],steps_per_epoch=v['steps_per_epoch'],peak_reserved_gib=v['peak_reserved_bytes']/2**30) for k,v in rows.items()}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),max_parallel=1,benchmark=cfg['benchmark'],fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS',json.dumps(cfg['benchmark']),flush=True)
def snapshot(cfg,active,phase):
    root=Path(cfg['runtime']);rows=[];known={}
    for sensor,arm,seed in cfg['queue']:
        out=root/'runs'/run_name(sensor,arm,seed);row=dict(sensor=sensor,arm=arm,seed=seed,training_complete=(out/'training_complete.json').exists(),evaluation_complete=(out/'evaluation_complete.json').exists())
        h=json.loads((out/'history.json').read_text(encoding='utf8')) if (out/'history.json').exists() else []
        d=[x['epoch_seconds'] for x in h if 'epoch_seconds' in x]
        if d:known.setdefault(f'{sensor}_{arm}',[]).extend(d[-3:])
        if (out/'status.json').exists():
            try:row['status']=json.loads((out/'status.json').read_text(encoding='utf8'))
            except (json.JSONDecodeError,PermissionError):pass
        rows.append(row)
    # Epoch seconds: measured recent epochs of the same sensor/arm, else benchmark step time x steps x 1.15 (validation and checkpoints).
    est={k:(float(np.median(known[k])) if k in known else v['median_step_seconds']*v['steps_per_epoch']*1.15) for k,v in cfg['benchmark'].items()}
    remaining=0.
    for row in rows:
        if row['training_complete']:continue
        st=row.get('status',{});done=max(0,st.get('epoch',1)-1+st.get('batch',0)/max(st.get('batches',1),1)) if st.get('phase')=='train' else 0
        remaining+=max(0,epochs(cfg,row['arm'])-done)*est[f"{row['sensor']}_{row['arm']}"]
    eta=datetime.now(KST)+timedelta(seconds=remaining)
    result=dict(time_kst=now(),phase=phase,active=[dict(sensor=s,arm=a,seed=x) for s,a,x in active],runs=rows,epoch_seconds=est,training_remaining_seconds=remaining,training_end_kst=eta.isoformat(),eta_basis='measured recent epochs per sensor/arm, else benchmark x1.15; excludes final RR/FR evaluation')
    write(root/'queue_status.json',result);return result

def run(resume=False):
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],resume=resume))
    def job(phase,sensor,arm,seed):
        out=root/'runs'/run_name(sensor,arm,seed);out.mkdir(parents=True,exist_ok=True);log=(out/(phase+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',phase,'--sensor',sensor,'--arm',arm,'--seed',str(seed)]
        if phase=='train' and resume and (out/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);print(json.dumps(dict(time_kst=now(),phase='launched_'+phase,sensor=sensor,arm=arm,seed=seed,pid=p.pid)),flush=True)
        while p.poll() is None:snapshot(cfg,[(sensor,arm,seed)],phase);time.sleep(5)
        log.close()
        if p.returncode:
            write(root/'controller_failure.json',dict(time_kst=now(),phase=phase,job=[sensor,arm,seed],returncode=p.returncode));raise RuntimeError(f'job failed {sensor} {arm} {seed}')
    for sensor,arm,seed in cfg['queue']:
        out=root/'runs'/run_name(sensor,arm,seed)
        if (out/'training_complete.json').exists():
            if json.loads((out/'training_complete.json').read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs')
        else:job('train',sensor,arm,seed)
        if not (out/'training_complete.json').exists():raise RuntimeError('training interrupted; refusing to advance the queue')
        # Test evaluation is deferred until all arms finish; selection remains validation-only.
    subprocess.run([sys.executable,'-u',str(HERE/'validation.py'),'--config',str(cp)],check=True)
    for sensor,arm,seed in cfg['queue']:
        out=root/'runs'/run_name(sensor,arm,seed)
        if not (out/'evaluation_complete.json').exists():job('evaluate',sensor,arm,seed)
    snapshot(cfg,[],'complete');write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(cfg['queue'])))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','optimize','run'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='optimize':optimize()
    else:run(a.resume)
