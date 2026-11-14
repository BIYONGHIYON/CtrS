"""Windows controller (from 80): QB, GF2 and WV3 caches rebuilt from the H5 and checked against the 63 (QB) and 65 stage 3 (GF2, WV3) hashes,
factorized initial weights made once from the 65 checkpoints (`factorize`, hashes sealed), sealed config, stage 1 (five arms on QB and WV3,
seed 42, two trainings at once) -> evaluation -> the validation-only stage gate of evaluate.stage_gate -> stage 2 (the chosen arm on QB,
GF2 and WV3, seeds 42-44) -> evaluation -> per-target 3-seed summary. Nothing is tuned between the stages."""
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
    perf=HERE.parent;root=perf.parents[1];e43=perf/'43_normalized_parallel_capacity';e54=perf/'54_multi_sensor_scale_mixed_training';e75=perf/'75_parallel_branch_width_depth';e58=perf/'58_spectral_loss_multi_sensor';e79=perf/'79_deep_narrow_branch_projection';e80=perf/'80_multi_sensor_deep_narrow_projection'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',e79/'model.py',e79/'prepare.py',e75/'model.py',e58/'train.py',e54/'model.py',e54/'prepare.py',e54/'train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),root/'common/interpolation.py',root/'common/metrics.py',root/'common/observation.py',root/'common/evaluation_protocol.py',root/'common/result_images.py']
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    paths += [root/p for p in conf['cache_reference_manifest'].values()]
    paths += [root/conf[key][sensor].replace('<seed>',str(s)) for key in ('controls','init_from','original') for sensor in conf['sensors'] for s in conf['seeds']]
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
    cfg['source_hashes']=source_hashes();cfg['created_kst']=now();cfg['coarse_cache_equal_to_reference']=matches
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cache='rebuilt from H5; normalized arrays equal the 63 (QB) and 65 stage 3 (GF2, WV3) hashes; coarse equality recorded',free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS',json.dumps(matches),flush=True)

def factorize():
    """Factorized initial weights for every sensor, seed and arm, from the same-seed 65 checkpoint (model.factorize): backbone SVD uses the
    second moment of the backbone inputs over CAL_PATCHES training patches drawn with a fixed generator; W2 CP-ALS is weight-only."""
    import torch
    import model as M
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot factorize after training initialized')
    dest=root/'init';dest.mkdir(exist_ok=True);hashes={};reports={}
    for sensor in cfg['sensors']:
        c={k:np.load(root/'cache'/sensor/f'train_{k}.npy',mmap_mode='r') for k in ('pan','lms','ms')}
        idx=np.sort(np.random.default_rng(M.CAL_SEED).choice(len(c['pan']),M.CAL_PATCHES,replace=False))
        cal={k:torch.from_numpy(np.array(c[k][idx])) for k in c};batches=[tuple(cal[k][i:i+32] for k in ('pan','lms','ms')) for i in range(0,M.CAL_PATCHES,32)]
        for seed in cfg['seeds']:
            src=HERE.parents[2]/cfg['init_from'][sensor].replace('<seed>',str(seed));ck=torch.load(src,map_location='cpu',weights_only=False)
            if ck['seed']!=seed:raise ValueError('seed mismatch '+str(src))
            base=M.M54.Model('C2MS',SENSORS[sensor]['bands'],seed);base.load_state_dict(ck['model'],strict=True)
            sigma=M.second_moments(base,batches,'cuda');base=base.cpu();made={}
            for arm in ARMS:
                key=M.SHAPES[arm]
                if key not in made:made[key]=M.factorize(base,arm,sigma)
                m,rep=made[key];target=M.Model(arm,SENSORS[sensor]['bands'],seed);target.load_state_dict(m.state_dict(),strict=True)
                p=dest/f'{sensor}_{arm}_s{seed}.pt';torch.save(dict(model=target.state_dict(),sensor=sensor,arm=arm,seed=seed,source=str(src.relative_to(HERE.parents[2])).replace('\\','/'),source_sha256=sha(src),report=rep,calibration_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest()),p)
                hashes[p.stem]=sha(p);reports[p.stem]=rep;print('FACTORIZED',p.stem,json.dumps(rep),flush=True)
    cfg['init_sha256']=hashes;cfg['factorization_report']=reports;write(cp,cfg);write(root/'factorization.json',dict(time_kst=now(),init_sha256=hashes,reports=reports))
    print('FACTORIZE PASS',len(hashes),flush=True)

def optimize():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    # Single-job step benchmark per sensor and arm; used only for the ETA (two jobs share the GPU, measured epochs replace it once known).
    rows={}
    for sensor in cfg['sensors']:
        for arm in ARMS:
            log=(out/f'benchmark_{sensor}_{arm}.log').open('w',encoding='utf8')
            code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--sensor',sensor,'--arm',arm,'--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
            if code:raise RuntimeError('benchmark failed '+sensor+' '+arm)
            rows[f'{sensor}_{arm}']=json.loads((out/f'benchmark_n1_{sensor}_{arm}.json').read_text(encoding='utf8'))
    import torch
    from model import Model,M54,conv_macs,used_parameters
    cfg['conv_gmac_256']={};cfg['used_parameters']={}
    for sensor in cfg['sensors']:
        nb=SENSORS[sensor]['bands'];shapes=[(1,1,256,256),(1,nb,256,256),(1,nb,64,64)];nets={**{arm:Model(arm,nb,42) for arm in ARMS},'C2MS':M54.Model('C2MS',nb,42),'C0':M54.Model('C0',nb,42)}
        cfg['conv_gmac_256'][sensor]={k:conv_macs(m,shapes)/1e9 for k,m in nets.items()};cfg['used_parameters'][sensor]={k:used_parameters(m) for k,m in nets.items()}
    cfg['max_parallel']=int(cfg['max_parallel']);cfg['benchmark']={k:dict(median_step_seconds=v['median_step_seconds'],steps_per_epoch=v['steps_per_epoch'],peak_reserved_gib=v['peak_reserved_bytes']/2**30) for k,v in rows.items()}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),max_parallel=cfg['max_parallel'],benchmark=cfg['benchmark'],conv_gmac_256=cfg['conv_gmac_256'],fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS',json.dumps(cfg['benchmark']),flush=True)
def snapshot(cfg,active,phase,queue):
    root=Path(cfg['runtime']);rows=[];known={}
    for sensor,arm,seed in queue:
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
    def launch(phase,sensor,arm,seed):
        out=root/'runs'/run_name(sensor,arm,seed);out.mkdir(parents=True,exist_ok=True);log=(out/(phase+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',phase,'--sensor',sensor,'--arm',arm,'--seed',str(seed)]
        if phase=='train' and resume and (out/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);print(json.dumps(dict(time_kst=now(),phase='launched_'+phase,sensor=sensor,arm=arm,seed=seed,pid=p.pid)),flush=True)
        return p,log
    def job(phase,sensor,arm,seed):
        p,log=launch(phase,sensor,arm,seed)
        while p.poll() is None:snapshot(cfg,[(sensor,arm,seed)],phase,[(sensor,arm,seed)]);time.sleep(5)
        log.close()
        if p.returncode:
            write(root/'controller_failure.json',dict(time_kst=now(),phase=phase,job=[sensor,arm,seed],returncode=p.returncode));raise RuntimeError(f'job failed {sensor} {arm} {seed}')
    def train_all(queue):
        pending=[]
        for sensor,arm,seed in queue:
            out=root/'runs'/run_name(sensor,arm,seed)
            if (out/'training_complete.json').exists():
                if json.loads((out/'training_complete.json').read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs')
            else:pending.append((sensor,arm,seed))
        running={}
        while pending or running:
            while pending and len(running)<cfg['max_parallel']:
                sensor,arm,seed=pending.pop(0);running[(sensor,arm,seed)]=launch('train',sensor,arm,seed)
            snapshot(cfg,list(running),'train',queue);time.sleep(5)
            for key,(p,log) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[key]
                if p.returncode or not (root/'runs'/run_name(*key)/'training_complete.json').exists():
                    for q,l in running.values():q.wait();l.close()
                    write(root/'controller_failure.json',dict(time_kst=now(),phase='train',job=list(key),returncode=p.returncode));raise RuntimeError(f'training failed {key}')
        for sensor,arm,seed in queue:
            if not (root/'runs'/run_name(sensor,arm,seed)/'evaluation_complete.json').exists():job('evaluate',sensor,arm,seed)
    stage1=[(s,a,cfg['latency_seed']) for s in cfg['stage1_sensors'] for a in ARMS];train_all(stage1)
    from evaluate import stage_gate,summarize
    gate=stage_gate(cfg);write(root/'stage_gate.json',gate);print(json.dumps(dict(time_kst=now(),phase='stage_gate',chosen=gate['chosen'])),flush=True)
    queue=list(stage1)
    if gate['chosen']:
        stage2=[(s,gate['chosen'],x) for x in cfg['seeds'] for s in cfg['sensors'] if (s,gate['chosen'],x) not in stage1];queue+=stage2;train_all(stage2)
        summarize(cfg,gate['chosen'],cfg['seeds'])
    snapshot(cfg,[],'complete',queue);write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(queue),stage_gate=gate['chosen']))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','factorize','optimize','run'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='factorize':factorize()
    elif a.mode=='optimize':optimize()
    else:run(a.resume)
