"""Windows controller (from 83): QB, GF2 and WV3 caches rebuilt from the H5 and checked against the 63 (QB) and 65 stage 3 (GF2, WV3) hashes
(`prepare`); stage 0 without training (`stage0`): reduced initial weights from the 65 checkpoints (hashes sealed), the fast K10 projection
checked against the 79/83 projection on every test scene of the 65 outputs, the fixed latency protocol and the speed gate; benchmark, sealed
config and preflight (`optimize`); stage 1 (FT0L and the arms passing the speed gate on QB and WV3, seed 42, two trainings at once) ->
evaluation -> the validation-only gate of evaluate.stage_gate -> stage 2 (the chosen arm on QB, GF2 and WV3, seeds 42-44) -> evaluation ->
per-target 3-seed summary -> final latency measurement (`run`). Nothing is tuned between the stages."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST,run_name,epochs
from prepare import SENSORS,data_files
from model import ARMS
from datetime import datetime,timedelta

PERF=HERE.parent;ROOT=PERF.parents[1]


def source_hashes():
    e43=PERF/'43_normalized_parallel_capacity';e54=PERF/'54_multi_sensor_scale_mixed_training'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',
           PERF/'83_factorized_c2ms_finetuning/model.py',PERF/'83_factorized_c2ms_finetuning/prepare.py',PERF/'79_deep_narrow_branch_projection/model.py',PERF/'79_deep_narrow_branch_projection/prepare.py',
           PERF/'75_parallel_branch_width_depth/model.py',PERF/'58_spectral_loss_multi_sensor/train.py',PERF/'82_cross_sensor_band_calibration/prepare.py',PERF/'82_cross_sensor_band_calibration/results/reference_stats/WV3.json',
           e54/'model.py',e54/'prepare.py',e54/'train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),
           ROOT/'common/interpolation.py',ROOT/'common/metrics.py',ROOT/'common/observation.py',ROOT/'common/evaluation_protocol.py',ROOT/'common/result_images.py']
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    paths+=[ROOT/p for p in conf['cache_reference_manifest'].values()]
    paths+=[ROOT/conf['init_from'][s].replace('<seed>',str(x)) for s in conf['sensors'] for x in conf['seeds']]
    paths+=[ROOT/conf['lite'][s].replace('<seed>',str(conf['latency_seed'])) for s in conf['latency_sensors']]
    paths=list(dict.fromkeys(paths))
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def runtime_cfg():
    conf=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(conf['runtime']);cp=root/'runtime_config.json'
    return root,cp,json.loads(cp.read_text(encoding='utf8'))


def prepare():
    # Rebuild the caches from the H5 as experiments 73-84 did, then compare every normalized array with the 63 (QB) and 65 stage 3 (GF2, WV3)
    # manifests (must match); coarse arrays are recorded (GPU/torch dependent).
    import torch,shutil
    from prepare import CoarsePairs
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True)
    if (root/'runtime_config.json').exists():raise RuntimeError('runtime already prepared; do not overwrite')
    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('less than 20 GiB free')
    old={s:json.loads((ROOT/p).read_text(encoding='utf8'))[s] for s,p in cfg['cache_reference_manifest'].items()};records={};matches={}
    for target in cfg['transfer_tests']:
        for split in ('RR','FR'):
            p=Path(cfg['data_root'])/data_files(target)[split];print('HASH',target,split,flush=True)
            if sha(p)!=cfg['transfer_data_sha256'][target][split]:raise ValueError('transfer data sha mismatch '+target+' '+split)
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


def projection_identity(cfg):
    """79/83 K10 projection vs prepare.FastProjector on the 65 outputs, every test scene (RR and FR) of every sensor, seed and transfer target.
    Pass: max |difference| <= tol_max_abs (normalized) on every scene and |mean RR PSNR difference| <= tol_psnr_db per (sensor, seed, target)."""
    import torch
    import evaluate as E
    from prepare import Projector,FastProjector
    from common.metrics import rr_metrics
    rule=cfg['projection_identity'];rows={};worst=0.;worst_psnr=0.
    for sensor in cfg['sensors']:
        for seed in cfg['seeds']:
            net,_=E.load(E.seeded(cfg,'init_from',sensor,seed),'L1S',SENSORS[sensor]['bands'],seed)
            for target in E.targets(sensor):
                ref,fast=Projector(target,max_iter=10),FastProjector(target,'cuda');mv=SENSORS[target]['max_value'];diffs=[];psnr=[[],[]];iters=[]
                for split in ('RR','FR'):
                    with h5py.File(Path(cfg['data_root'])/data_files(target)[split]) as f:
                        for i in range(len(f['pan'])):
                            x=[torch.from_numpy((np.asarray(f[k][i],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]
                            with torch.inference_mode():y=net(*x).double()
                            with torch.inference_mode(False):a,ia=ref(y.clone(),x[2].double().clone())
                            with torch.inference_mode():b,ib=fast(y,x[2].double())
                            diffs.append(float((a.detach()-b).abs().max()));iters.append([ia,ib])
                            if split=='RR':
                                gt=np.asarray(f['gt'][i],dtype=np.float64)
                                for j,o in enumerate((a.detach(),b)):psnr[j].append(rr_metrics(gt,o[0].float().cpu().numpy().astype(np.float64)*mv)['PSNR'])
                d=float(np.mean(psnr[1])-np.mean(psnr[0]));worst=max(worst,max(diffs));worst_psnr=max(worst_psnr,abs(d))
                rows[f'{sensor}_s{seed}_{target}']=dict(max_abs=max(diffs),mean_rr_psnr_ref=float(np.mean(psnr[0])),mean_rr_psnr_fast=float(np.mean(psnr[1])),psnr_diff_db=d,iterations_equal=all(p==q for p,q in iters))
                print(json.dumps(dict(time_kst=now(),identity=f'{sensor}_s{seed}_{target}',max_abs=max(diffs),psnr_diff_db=d)),flush=True)
    passed=worst<=rule['tol_max_abs'] and worst_psnr<=rule['tol_psnr_db'] and all(r['iterations_equal'] for r in rows.values())
    return dict(time_kst=now(),rule=rule,rows=rows,worst_max_abs=worst,worst_psnr_diff_db=worst_psnr,passed=passed)


def latency_systems(cfg,arm_weights):
    """{sensor: {name: module}} for the latency protocol. arm_weights(sensor, arm) -> path of the arm weights to time."""
    import evaluate as E
    from prepare import Projected,FastProjected
    out={};seed=cfg['latency_seed']
    for sensor in cfg['latency_sensors']:
        nb=SENSORS[sensor]['bands'];l1,_=E.load(E.seeded(cfg,'init_from',sensor,seed),'L1S',nb,seed);k2,_=E.load(E.seeded(cfg,'lite',sensor,seed),'K2',nb,seed)
        s={'C2MS':l1,'C2MS-SP (79 projection)':Projected(l1,sensor),'C2MS-SP (fast projection)':FastProjected(l1,sensor),'Lite K2':k2,'Lite K2 + fast projection':FastProjected(k2,sensor)}
        for arm,path in arm_weights(sensor).items():
            net,_=E.load(path,arm,nb,seed);s[arm]=net;s[arm+' + fast projection']=FastProjected(net,sensor)
        out[sensor]=s
    return out


def stage0():
    """No training: reduced initial weights (sealed), projection identity, stage-0 latency and the speed gate (fixed before training)."""
    import torch
    import model as M
    import evaluate as E
    root,cp,cfg=runtime_cfg()
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot run stage 0 after training initialized')
    E.T.setup(42);dest=root/'init';dest.mkdir(exist_ok=True);hashes={};reports={}
    for sensor in cfg['sensors']:
        c={k:np.load(root/'cache'/sensor/f'train_{k}.npy',mmap_mode='r') for k in ('pan','lms','ms')}
        idx=np.sort(np.random.default_rng(M.CAL_SEED).choice(len(c['pan']),M.CAL_PATCHES,replace=False))
        cal={k:torch.from_numpy(np.array(c[k][idx])) for k in c};batches=[tuple(cal[k][i:i+32] for k in ('pan','lms','ms')) for i in range(0,M.CAL_PATCHES,32)]
        for seed in cfg['seeds']:
            src=E.seeded(cfg,'init_from',sensor,seed);ck=torch.load(src,map_location='cpu',weights_only=False)
            if ck['seed']!=seed:raise ValueError('seed mismatch '+str(src))
            base=M.M54.Model('C2MS',SENSORS[sensor]['bands'],seed);base.load_state_dict(ck['model'],strict=True)
            for arm in ARMS:
                m,rep=M.reduce(base,arm,batches,'cuda');target=M.Model(arm,SENSORS[sensor]['bands'],seed);target.load_state_dict(m.state_dict(),strict=True)
                p=dest/f'{sensor}_{arm}_s{seed}.pt';torch.save(dict(model=target.state_dict(),sensor=sensor,arm=arm,seed=seed,source=str(src.relative_to(ROOT)).replace('\\','/'),source_sha256=sha(src),report=rep,calibration_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest()),p)
                hashes[p.stem]=sha(p);reports[p.stem]=rep;print('REDUCED',p.stem,json.dumps(rep),flush=True)
    identity=projection_identity(cfg);write(root/'projection_identity.json',identity);print('PROJECTION IDENTITY',identity['passed'],identity['worst_max_abs'],identity['worst_psnr_diff_db'],flush=True)
    timing=E.measure(cfg,latency_systems(cfg,lambda s:{a:dest/f'{s}_{a}_s{cfg["latency_seed"]}.pt' for a in ARMS if a!='FT0L'}),'stage0');write(root/'stage0_timing.json',timing)
    base={k:timing['result'][f'{k}_RR_256']['C2MS']['seconds'] for k in cfg['latency_sensors']};ratio={}
    for arm in ARMS:
        if arm=='FT0L':continue
        ratio[arm]={k:timing['result'][f'{k}_RR_256'][arm]['seconds']/base[k] for k in cfg['latency_sensors']}
    fast=[a for a,r in ratio.items() if all(v<=cfg['speed_gate']['max_ratio'] for v in r.values())]
    order=sorted(fast,key=lambda a:float(np.mean(list(ratio[a].values()))))
    cfg['init_sha256']=hashes;cfg['reduction_report']=reports;cfg['projection']='fast' if identity['passed'] else 'reference'
    cfg['speed_gate']['network_time_ratio_vs_C2MS']=ratio;cfg['trained_arms']=['FT0L']+order;cfg['gate']['order']=order
    write(cp,cfg);write(root/'stage0.json',dict(time_kst=now(),projection=cfg['projection'],speed_ratio=ratio,trained_arms=cfg['trained_arms'],gate_order=order,init_sha256=hashes,reports=reports))
    print('STAGE0 PASS',json.dumps(dict(projection=cfg['projection'],trained=cfg['trained_arms'],ratio=ratio)),flush=True)


def optimize():
    root,cp,cfg=runtime_cfg();out=root/'preflight';out.mkdir(exist_ok=True)
    if 'trained_arms' not in cfg:raise RuntimeError('stage 0 missing')
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    rows={}
    for sensor in cfg['sensors']:
        for arm in cfg['trained_arms']:
            log=(out/f'benchmark_{sensor}_{arm}.log').open('w',encoding='utf8')
            code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','bench','--sensor',sensor,'--arm',arm,'--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
            if code:raise RuntimeError('benchmark failed '+sensor+' '+arm)
            rows[f'{sensor}_{arm}']=json.loads((out/f'benchmark_n1_{sensor}_{arm}.json').read_text(encoding='utf8'))
    from model import Model,M54,conv_macs,used_parameters
    cfg['conv_gmac_256']={};cfg['used_parameters']={}
    for sensor in cfg['sensors']:
        nb=SENSORS[sensor]['bands'];shapes=[(1,1,256,256),(1,nb,256,256),(1,nb,64,64)];nets={**{arm:Model(arm,nb,42) for arm in ARMS},'C2MS':M54.Model('C2MS',nb,42)}
        cfg['conv_gmac_256'][sensor]={k:conv_macs(m,shapes)/1e9 for k,m in nets.items()};cfg['used_parameters'][sensor]={k:used_parameters(m) for k,m in nets.items()}
    cfg['max_parallel']=int(cfg['max_parallel']);cfg['benchmark']={k:dict(median_step_seconds=v['median_step_seconds'],steps_per_epoch=v['steps_per_epoch'],peak_reserved_gib=v['peak_reserved_bytes']/2**30) for k,v in rows.items()}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),max_parallel=cfg['max_parallel'],benchmark=cfg['benchmark'],conv_gmac_256=cfg['conv_gmac_256'],used_parameters=cfg['used_parameters'],fingerprint=cfg['fingerprint']))
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
    est={k:(float(np.median(known[k])) if k in known else v['median_step_seconds']*v['steps_per_epoch']*1.15) for k,v in cfg['benchmark'].items()}
    remaining=0.
    for row in rows:
        if row['training_complete']:continue
        st=row.get('status',{});done=max(0,st.get('epoch',1)-1+st.get('batch',0)/max(st.get('batches',1),1)) if st.get('phase')=='train' else 0
        remaining+=max(0,epochs(cfg,row['arm'])-done)*est[f"{row['sensor']}_{row['arm']}"]
    eta=datetime.now(KST)+timedelta(seconds=remaining)
    result=dict(time_kst=now(),phase=phase,active=[dict(sensor=s,arm=a,seed=x) for s,a,x in active],runs=rows,epoch_seconds=est,training_remaining_seconds=remaining,training_end_kst=eta.isoformat(),
                eta_basis='sum of the remaining epochs x recent epoch seconds of every queued run; two runs share the GPU, so the real end is earlier; excludes evaluation')
    write(root/'queue_status.json',result);return result


def final_measure(cfg,chosen):
    """After training: the latency protocol with the trained weights (seed 42), plus reference settings (cuDNN autotune + TF32, FP16 autocast)
    and the FP16 RR change of C2MS-SP and the timed arms on the training sensors (reference only, never a judgement)."""
    import torch
    import evaluate as E
    root=Path(cfg['runtime']);arms=[chosen] if chosen else [a for a in cfg['trained_arms'] if a!='FT0L']
    weights=lambda s:{a:root/'runs'/run_name(s,a,cfg['latency_seed'])/'checkpoints/best.pt' for a in arms}
    out=dict(official=E.measure(cfg,latency_systems(cfg,weights),'final'))
    out['optimized_reference']=E.measure(cfg,latency_systems(cfg,weights),'final optimized',setting='optimized')
    out['fp16_reference']=E.measure(cfg,latency_systems(cfg,weights),'final fp16',fp16=True)
    E.T.setup(cfg['latency_seed']);fp16={}
    for sensor in cfg['latency_sensors']:
        sys_=latency_systems(cfg,weights)[sensor];mv=SENSORS[sensor]['max_value'];fp16[sensor]={}
        for name in ['C2MS-SP (fast projection)']+[a+' + fast projection' for a in arms]:
            vals={False:[],True:[]}
            with h5py.File(Path(cfg['data_root'])/data_files(sensor)['RR']) as f:
                for i in range(len(f['pan'])):
                    a={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')}
                    for half in (False,True):
                        with torch.autocast('cuda',dtype=torch.float16,enabled=half):vals[half].append(E.rr_metrics(a['gt'],E.V.predict(sys_[name],a['pan'],a['lms'],a['ms'],mv,'cuda')))
            fp16[sensor][name]={k:float(np.mean([x[k] for x in vals[True]])-np.mean([x[k] for x in vals[False]])) for k in ('PSNR','SAM','ERGAS')}
    out['fp16_rr_change_reference']=fp16;write(root/'final_timing.json',out);return out


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
    stage1=[(s,a,cfg['latency_seed']) for s in cfg['stage1_sensors'] for a in cfg['trained_arms']];train_all(stage1)
    from evaluate import stage_gate,summarize
    gate=stage_gate(cfg);write(root/'stage_gate.json',gate);print(json.dumps(dict(time_kst=now(),phase='stage_gate',chosen=gate['chosen'])),flush=True)
    queue=list(stage1)
    if gate['chosen']:
        stage2=[(s,gate['chosen'],x) for x in cfg['seeds'] for s in cfg['sensors'] if (s,gate['chosen'],x) not in stage1];queue+=stage2;train_all(stage2)
        summarize(cfg,gate['chosen'],cfg['seeds'])
    snapshot(cfg,[],'final timing',queue);final_measure(cfg,gate['chosen'])
    snapshot(cfg,[],'complete',queue);write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(queue),stage_gate=gate['chosen']))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='85 controller: prepare -> stage0 -> optimize -> run (resume with --mode run --resume)')
    p.add_argument('--mode',choices=['prepare','stage0','optimize','run'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    {'prepare':prepare,'stage0':stage0,'optimize':optimize}.get(a.mode,lambda:run(a.resume))()
