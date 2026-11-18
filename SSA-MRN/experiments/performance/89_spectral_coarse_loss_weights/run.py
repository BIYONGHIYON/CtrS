"""89 Windows controller. Modes (chain89.ps1 runs them in this order and stops at the first nonzero exit):

  gate88   : stage 0 check (CPU only). The 88 seed-42 run is complete (all_complete.json, no controller_failure.json), its summary exists and
             the K2D evaluation of QB, GF2 and WV3 is finite. K2D is taken as adopted (user decision 2026-11-19 04h); no performance gate.
  prepare  : QB, GF2, WV3 caches rebuilt from the H5 and checked against the 63 / 65 stage 3 hashes (as 88)
  optimize : on the idle GPU: W2 output implementation timing and choice, controls (C2MS-SP + P and 88 K2D + P on RR test), parameter
             and factorization-target checks, benchmark, sealed config, 58 preflight
  run      : two jobs at once. W2T (QB) -> factorize -> W2K (QB) = the W2 2x2 check, together with the stage-A screening runs (shared points
             first). Each run is evaluated (RR test, report only) right after its training. Stage B starts after every stage-A run is trained
             and evaluated, the W2 check passed and the stage-A selection (validation, selection.py) chose lambda*; after stage B the c
             selection is written and the controller ends (the confirmation stage is not scheduled).
             Stop rule (user decision): on any failure - a process error, a run that is not complete, a validation sanity failure
             (best epoch 0, nonfinite, or validation PSNR more than `sanity_margin_db` below the same-seed C2MS-SP), or a W2 check below
             -0.06 dB - no new job is launched, running jobs finish normally (never killed), stopped.json is written and the exit code is 3.
  factorize: K2 initial weights of W2K from the hash-checked W2T epoch-120 snapshot (83 factorization, 83 calibration patches)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST,run_name,epochs,set_impl
from prepare import SENSORS,data_files
import model as M
from datetime import datetime,timedelta

PERF=HERE.parent;ROOT=PERF.parents[1]
SEED=42


def load_conf():return json.loads((HERE/'config.json').read_text(encoding='utf8'))


def source_hashes():
    e43=PERF/'43_normalized_parallel_capacity';e54=PERF/'54_multi_sensor_scale_mixed_training';e88=PERF/'88_factorization_at_decay_start'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',HERE/'selection.py',e88/'model.py',e88/'prepare.py',
           PERF/'86_fft_adjoint_fast_inference/prepare.py',PERF/'85_hidden_width_reduction_fast_projection/prepare.py',PERF/'83_factorized_c2ms_finetuning/model.py',PERF/'83_factorized_c2ms_finetuning/prepare.py',PERF/'79_deep_narrow_branch_projection/model.py',PERF/'79_deep_narrow_branch_projection/prepare.py',
           PERF/'75_parallel_branch_width_depth/model.py',PERF/'58_spectral_loss_multi_sensor/train.py',PERF/'82_cross_sensor_band_calibration/prepare.py',
           e54/'model.py',e54/'prepare.py',e54/'train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),
           ROOT/'common/interpolation.py',ROOT/'common/metrics.py',ROOT/'common/observation.py',ROOT/'common/evaluation_protocol.py']
    conf=load_conf()
    paths+=[ROOT/p for p in conf['cache_reference_manifest'].values()]
    paths+=[ROOT/conf['controls']['c2ms_sp'][s].replace('<seed>',str(SEED)) for s in conf['sensors']]
    paths=list(dict.fromkeys(paths))
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def runtime_cfg():
    conf=load_conf();root=Path(conf['runtime']);cp=root/'runtime_config.json'
    return root,cp,json.loads(cp.read_text(encoding='utf8'))


def gate88():
    """Stage 0: the 88 seed-42 run finished without failure and the K2D evaluation is finite (adoption decided by the user)."""
    conf=load_conf();r=Path(conf['stage0']['runtime']);out=Path(conf['runtime']).parent;out.mkdir(parents=True,exist_ok=True);problems=[]
    if not (r/'all_complete.json').exists():problems.append('88 all_complete.json missing')
    if (r/'controller_failure.json').exists():problems.append('88 controller_failure.json present')
    rec=dict(time_kst=now(),rule=conf['stage0']['gate'],runtime=str(r))
    if not problems:
        s=json.loads((r/'summary.json').read_text(encoding='utf8'));k=s['arms']['K2D'];rec['summary_sha256']=sha(r/'summary.json')
        rec['official_label']=s.get('official_label');rec['light_non_inferior']=k.get('light_non_inferior')
        rec['delta_RR_vs_C2MS_SP']={t:v['judgement']['arm_P_vs_C2MS_SP']['delta_RR'] for t,v in k['targets'].items()}
        for sensor in conf['sensors']:
            e=r/'runs'/f'{sensor}_K2D_s{SEED}'/'evaluation_complete.json'
            if not e.exists():problems.append('88 K2D evaluation missing '+sensor);continue
            d=rec['delta_RR_vs_C2MS_SP'].get(sensor,{})
            if not d or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in d.values()):problems.append('88 K2D RR not finite '+sensor)
            if not Path(conf['controls']['k2d_88'][sensor].replace('<seed>',str(SEED))).exists():problems.append('88 K2D best.pt missing '+sensor)
    rec['problems']=problems;rec['pass']=not problems;write(out/'gate88.json',rec)
    print('GATE88','PASS' if rec['pass'] else 'FAIL',json.dumps(problems),json.dumps(rec.get('delta_RR_vs_C2MS_SP')),flush=True);sys.exit(0 if rec['pass'] else 3)


def prepare():
    import torch,shutil
    from prepare import CoarsePairs
    cfg=load_conf();root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True)
    if (root/'runtime_config.json').exists():raise RuntimeError('runtime already prepared; do not overwrite')
    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('less than 20 GiB free')
    old={s:json.loads((ROOT/p).read_text(encoding='utf8'))[s] for s,p in cfg['cache_reference_manifest'].items()};records={};matches={}
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
                        rec['coarse'][k]=dict(shape=list(shapes[k]),sha256=h);matches[f'{sensor}_{split}_coarse_{k}']=h==old[sensor][split]['coarse'][k]['sha256']
                    print('CACHE',sensor,split,flush=True)
            records[sensor][split]=rec
    cfg['source_hashes']=source_hashes();cfg['created_kst']=now();cfg['coarse_cache_equal_to_reference']=matches
    write(root/'data_manifest.json',records);cfg['cache_manifest_sha256']=sha(root/'data_manifest.json');write(root/'runtime_config.json',cfg)
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS',json.dumps(matches),flush=True)


def structure_checks():
    """2x2 conversion keeps the function (random mapped weights), K2 targets exclude the 2x2 layers, W2K has 192 fewer parameters than K2."""
    import torch
    out={}
    for impl in ('convT','pixelshuffle'):
        torch.manual_seed(5);orig=M.M54.Model('C2MS',4,42);br=orig.core.backbone.branch
        with torch.no_grad():br[-2].weight.copy_(torch.randn_like(br[-2].weight)*.05)  # bias kept 0: 2x2 has per-channel bias only
        new=M.to_2x2(M.M54.Model('C2MS',4,42),impl)
        with torch.no_grad():M.set_out_weight(new.core.backbone.branch[-1],br[-2].weight.detach().reshape(br[-2].out_channels,-1))
        x=[torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16)]
        with torch.inference_mode():d=float((orig.eval()(*x)-new.eval()(*x)).abs().max())
        if d>1e-5:raise AssertionError(f'2x2 conversion changes the output ({impl}): {d}')
        out[f'conversion_max_abs_{impl}']=d
    w2t=M.Model('W2T',4,42);names=[n for n,_p,_k,g in M.M83.targets(w2t) if g=='w2']
    if len(names)!=16:raise AssertionError('expected 16 W2 3x3 targets, got %d'%len(names))
    for n,parent,key,g in M.M83.targets(w2t):
        c=M.M83._get(parent,key)
        if not (isinstance(c,torch.nn.Conv2d) and c.kernel_size==(3,3)):raise AssertionError('factorization target is not a 3x3 conv: '+n)
    p=lambda m:sum(x.numel() for x in m.parameters())
    out['parameters']=dict(C2MS=p(M.Model('C2MS',4,42)),W2T=p(w2t),K2=p(M.Model('K2',4,42)),W2K=p(M.Model('W2K',4,42)))
    if out['parameters']['K2']-out['parameters']['W2K']!=192 or out['parameters']['C2MS']-out['parameters']['W2T']!=192:raise AssertionError('parameter difference is not 192: '+json.dumps(out['parameters']))
    return out


def factorize(sensor,arm,seed):
    import torch
    import train as TR
    root,cp,cfg=runtime_cfg();set_impl(cfg);dest=root/'init';dest.mkdir(exist_ok=True);p=TR.init_path(cfg,sensor,arm,seed);rec=p.with_suffix('.json')
    if rec.exists():print('FACTORIZED ALREADY',p.stem,flush=True);return
    k,snap,meta=TR.snapshot_files(cfg,sensor,seed);src=snap/'latest.pt';ck=torch.load(src,map_location='cpu',weights_only=False)
    if ck['epoch']!=k or ck['fingerprint']!=cfg['fingerprint'] or ck['seed']!=seed:raise RuntimeError('snapshot is not the epoch-%d state'%k)
    source=dict(run=run_name(sensor,M.SOURCE[arm],seed),file='snapshots/e%d/latest.pt'%k,epoch=k,sha256=meta['files']['latest.pt']['sha256'])
    TR.T.setup(seed)
    c={k_:np.load(root/'cache'/sensor/f'train_{k_}.npy',mmap_mode='r') for k_ in ('pan','lms','ms')}
    idx=np.sort(np.random.default_rng(M.CAL_SEED).choice(len(c['pan']),M.CAL_PATCHES,replace=False))
    cal={k_:torch.from_numpy(np.array(c[k_][idx])) for k_ in c};batches=[tuple(cal[k_][i:i+32] for k_ in ('pan','lms','ms')) for i in range(0,M.CAL_PATCHES,32)]
    base=M.Model(M.SOURCE[arm],SENSORS[sensor]['bands'],seed);base.load_state_dict(ck['model'],strict=True)
    sigma=M.second_moments(base,batches,'cuda');base=base.cpu();m,rep=M.factorize(base,'K2',sigma)
    target=M.Model(arm,SENSORS[sensor]['bands'],seed);target.load_state_dict(m.state_dict(),strict=True)
    tmp=p.with_name(p.stem+'_building.pt');torch.save(dict(model=target.state_dict(),sensor=sensor,arm=arm,seed=seed,source=source,report=rep,calibration_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest()),tmp);os.replace(tmp,p)
    write(rec,dict(time_kst=now(),sha256=sha(p),bytes=p.stat().st_size,source=source,report=rep,calibration_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest()))
    print('FACTORIZED',p.stem,json.dumps(rep),flush=True)


def optimize():
    import torch
    import evaluate as E
    root,cp,cfg=runtime_cfg();out=root/'preflight';out.mkdir(exist_ok=True)
    if any((root/'runs').glob('*/checkpoints/initial.pt')):raise RuntimeError('cannot optimize after training initialized')
    E.setup(SEED);timing=E.measure_out_impl(cfg);write(out/'w2_out_impl_timing.json',timing);cfg['w2_out_impl']=timing['choice'];set_impl(cfg)
    cfg['structure_checks']=structure_checks()
    for sensor in cfg['sensors']:E.control_rows(cfg,sensor,SEED)
    cfg['controls_sha256']={s:sha(root/'controls'/f'{s}_s{SEED}.json') for s in cfg['sensors']}
    rows={}
    for sensor in cfg['sensors']:
        for arm in (('W2T','W2K') if sensor=='QB' else ('W2T',)):
            log=(out/f'benchmark_{sensor}_{arm}.log').open('w',encoding='utf8')
            cfg_tmp=dict(cfg,fingerprint='benchmark');write(out/'bench_config.json',cfg_tmp)
            code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(out/'bench_config.json'),'--mode','bench','--sensor',sensor,'--arm',arm,'--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
            if code:raise RuntimeError('benchmark failed '+sensor+' '+arm)
            b=json.loads((out/f'benchmark_n1_{sensor}_{arm}.json').read_text(encoding='utf8'));rows[f'{sensor}_{arm}']=dict(median_step_seconds=b['median_step_seconds'],steps_per_epoch=b['steps_per_epoch'],peak_reserved_gib=b['peak_reserved_bytes']/2**30)
    cfg['benchmark']=rows
    shapes=lambda nb:[(1,1,256,256),(1,nb,256,256),(1,nb,64,64)]
    cfg['gmac_256']={s:{a:E.macs(M.Model(a,SENSORS[s]['bands'],42),shapes(SENSORS[s]['bands']))/1e9 for a in ('C2MS','W2T','K2','W2K')} for s in cfg['sensors']}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),w2_out_impl=cfg['w2_out_impl'],structure_checks=cfg['structure_checks'],benchmark=rows,gmac_256=cfg['gmac_256'],fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS',cfg['w2_out_impl'],json.dumps(rows),flush=True)


def stage_runs(cfg,stage,lam_star=None):
    st=cfg['stage_'+stage];shared=[float(x) for x in st['shared_points']];keys=[]
    mk=(lambda v:M.screen_arm(v,1.0)) if stage=='A' else (lambda v:M.screen_arm(lam_star,v,'B'))
    for v in shared:
        for s in ('GF2','WV3','QB'):
            if v in [float(x) for x in st['candidates'][s]]:keys.append((s,mk(v),SEED))
    keys+=[('QB',mk(float(v)),SEED) for v in st['candidates']['QB'] if float(v) not in shared]
    return list(dict.fromkeys(keys))


def validation_of(root,key):
    d=json.loads((root/'runs'/run_name(*key)/'training_complete.json').read_text(encoding='utf8'))
    return dict(psnr=d['best_val']['val_psnr'],sam=d['best_val']['val_sam'],mse=d['best_val']['val_mse'],epoch=d['best_epoch'])


def sanity(cfg,root,key):
    """Validation sanity of a finished run: best epoch > 0, finite, PSNR not more than sanity_margin_db below the same-seed C2MS-SP."""
    import torch
    v=validation_of(root,key);ref=cfg['_ref_val'][key[0]]
    problems=[]
    if not all(math.isfinite(x) for x in (v['psnr'],v['sam'],v['mse'])):problems.append('nonfinite validation')
    if v['epoch']<1:problems.append('best epoch is 0 (no improvement over the initial state)')
    if v['psnr']<ref-cfg['sanity_margin_db']:problems.append('validation PSNR %.3f is more than %.2f dB below C2MS-SP %.3f'%(v['psnr'],cfg['sanity_margin_db'],ref))
    return problems


def w2_check(cfg,root):
    v=validation_of(root,('QB','W2K',SEED));c=json.loads(Path(cfg['controls']['k2d_88_complete']['QB'].replace('<seed>',str(SEED))).read_text(encoding='utf8'))
    d=v['psnr']-c['best_val']['val_psnr'];ev=root/'runs'/run_name('QB','W2K',SEED)/'evaluation_complete.json'
    rec=dict(time_kst=now(),rule=cfg['w2_check']['apply_rule'],w2k_validation=v,k2d_88_validation=c['best_val'],k2d_88_best_epoch=c['best_epoch'],delta_val_psnr=d,
             delta_val_sam=v['sam']-c['best_val']['val_sam'],apply=d>=-0.06,
             test_report_only=json.loads(ev.read_text(encoding='utf8'))['delta_RR'] if ev.exists() else None,w2_out_impl=cfg['w2_out_impl'])
    write(root/'w2_check.json',rec);return rec


def snapshot(cfg,root,running,phase,keys,stop_reason=None):
    rows=[];known={}
    for key in keys:
        out=root/'runs'/run_name(*key);r=dict(sensor=key[0],arm=key[1],trained=(out/'training_complete.json').exists(),evaluated=(out/'evaluation_complete.json').exists())
        if (out/'status.json').exists():
            try:st=json.loads((out/'status.json').read_text(encoding='utf8'));r.update(phase=st.get('phase'),epoch=st.get('epoch'),epochs=st.get('epochs'))
            except (json.JSONDecodeError,PermissionError):pass
        if r['trained']:
            try:r['validation']=validation_of(root,key)
            except Exception:pass
        rows.append(r)
    write(root/'queue_status.json',dict(time_kst=now(),phase=phase,running=[dict(job=j,sensor=k[0],arm=k[1]) for (j,k) in running],stop_reason=stop_reason,runs=rows))


def run(resume=False):
    conf=load_conf();root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));set_impl(cfg)
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    import torch
    cfg['_ref_val']={s:torch.load(ROOT/cfg['controls']['c2ms_sp'][s].replace('<seed>',str(SEED)),map_location='cpu',weights_only=False)['metrics']['val_psnr'] for s in cfg['sensors']}
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],resume=resume))
    def done(key,what):
        out=root/'runs'/run_name(*key)
        if what=='train':
            f=out/'training_complete.json'
            if not f.exists():return False
            if json.loads(f.read_text(encoding='utf8'))['fingerprint']!=cfg['fingerprint']:raise RuntimeError('completion contract differs '+str(f))
            return True
        return (out/'evaluation_complete.json').exists()
    def launch(phase,key):
        sensor,arm,seed=key;out=root/'runs'/run_name(*key);out.mkdir(parents=True,exist_ok=True);log=(out/(phase+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',phase,'--sensor',sensor,'--arm',arm,'--seed',str(seed)]
        if phase=='train' and resume and (out/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);print(json.dumps(dict(time_kst=now(),launched=phase,sensor=sensor,arm=arm,pid=p.pid)),flush=True);return p,log
    def factorize_job(key):
        log=root/'init'/f'factorize_{run_name(*key)}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf8') as f:code=subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'--mode','factorize','--sensor',key[0],'--arm',key[1],'--seed',str(key[2])],stdout=f,stderr=subprocess.STDOUT).returncode
        return code
    state=dict(stop=None)
    def stop(reason):
        if state['stop'] is None:state['stop']=reason;print(json.dumps(dict(time_kst=now(),stop=reason)),flush=True)
    def execute(keys,extra_ready=None,on_eval=None,phase='train'):
        """Train then evaluate every key (two jobs at once). extra_ready(key) gates dependent keys. Returns when all are done or stopped."""
        jobs=[];running={}
        for k in keys:
            if not done(k,'train'):jobs.append(('train',k))
            elif not done(k,'eval'):
                prob=sanity(cfg,root,k)
                if prob:stop(f'{run_name(*k)}: '+'; '.join(prob))
                jobs.append(('evaluate',k))
        while True:
            if state['stop'] is None:
                for j in sorted(jobs,key=lambda j:(j[0]!='evaluate',j[1][1]!='W2K')):
                    if len(running)>=cfg['max_parallel']:break
                    kind,k=j
                    if kind=='train' and extra_ready and not extra_ready(k):continue
                    if kind=='train' and k[1] in M.FACTORIZED:
                        if factorize_job(k):stop('factorize failed '+run_name(*k));break
                    jobs.remove(j);running[j]=launch(kind,k)
            snapshot(cfg,root,list(running),phase,keys,state['stop']);time.sleep(5)
            for j,(p,log) in list(running.items()):
                if p.poll() is None:continue
                log.close();del running[j];kind,k=j
                if p.returncode or (kind=='train' and not done(k,'train')) or (kind=='evaluate' and not done(k,'eval')):
                    stop(f'{kind} failed {run_name(*k)} (exit {p.returncode})');continue
                if kind=='train':
                    prob=sanity(cfg,root,k)
                    if prob:stop(f'{run_name(*k)}: '+'; '.join(prob));continue
                    jobs.append(('evaluate',k))
                elif on_eval:on_eval(k)
            if not running and (state['stop'] is not None or not jobs):break
            if not running and jobs and all(j[0]=='train' and extra_ready and not extra_ready(j[1]) for j in jobs):stop('no runnable job: '+str(jobs));break
        snapshot(cfg,root,[],phase,keys,state['stop'])
    def finish_stopped():
        write(root/'stopped.json',dict(time_kst=now(),reason=state['stop'],note='no new job was launched after the problem; running jobs finished normally'))
        print('STOPPED',state['stop'],flush=True);sys.exit(3)
    # stage W + A: the W2 2x2 check (W2T -> W2K) together with the stage-A screening runs
    w2=[('QB','W2T',SEED),('QB','W2K',SEED)];A=stage_runs(cfg,'A')
    ready=lambda k:k[1]!='W2K' or done(('QB','W2T',SEED),'train')
    def on_eval(k):
        if k[1]=='W2K':
            r=w2_check(cfg,root);print(json.dumps(dict(time_kst=now(),w2_check=r['delta_val_psnr'],apply=r['apply'])),flush=True)
            if not r['apply']:stop('W2 2x2 check: validation PSNR %.4f dB below -0.06 dB'%r['delta_val_psnr'])
    if done(('QB','W2K',SEED),'eval'):on_eval(('QB','W2K',SEED))
    execute(w2+A,ready,on_eval,'W2 check + stage A')
    if state['stop']:finish_stopped()
    import selection as S
    valA={s:{M.parse(k[1])[0]:validation_of(root,k) for k in A if k[0]==s} for s in cfg['sensors']}
    selA=S.choose('A',cfg,valA);selA['time_kst']=now();selA['screening_check']=dict(qb_lambda0_delta_psnr=selA['delta_psnr']['QB'].get(0.0),qb_lambda0_delta_sam=selA['delta_sam']['QB'].get(0.0),reference_150_epochs=cfg['screening_check']['compare'])
    write(root/'selection_A.json',selA);lam=selA['chosen'];print(json.dumps(dict(time_kst=now(),selection_A=lam,at_boundary=selA['at_boundary'],fallback=selA['fell_back_to_reference'])),flush=True)
    B=stage_runs(cfg,'B',lam)
    execute(B,None,None,'stage B (lambda %.2f)'%lam)
    if state['stop']:finish_stopped()
    valB={s:{M.parse(k[1])[1]:validation_of(root,k) for k in B if k[0]==s} for s in cfg['sensors']}
    selB=S.choose('B',cfg,valB);selB['time_kst']=now();selB['lambda_star']=lam
    a_same=('QB',M.screen_arm(lam,1.0),SEED)
    if (root/'runs'/run_name(*a_same)/'training_complete.json').exists():
        selB['rerun_spread_QB']=dict(stage_A=validation_of(root,a_same),stage_B=validation_of(root,('QB',M.screen_arm(lam,1.0,'B'),SEED)))
    write(root/'selection_B.json',selB)
    write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],lambda_star=lam,c_star=selB['chosen'],runs=len(w2)+len(A)+len(B),next='confirmation (K2D 150 epochs x QB, GF2, WV3) after the user decision'))
    print('ALL COMPLETE lambda* %.2f c* %.2f'%(lam,selB['chosen']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='89 controller: gate88 -> prepare -> optimize -> run (resume: --mode run --resume)')
    p.add_argument('--mode',choices=['gate88','prepare','optimize','run','factorize'],required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--sensor');p.add_argument('--arm');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.mode=='factorize':factorize(a.sensor,a.arm,a.seed)
    else:{'gate88':gate88,'prepare':prepare,'optimize':optimize}.get(a.mode,lambda:run(a.resume))()
