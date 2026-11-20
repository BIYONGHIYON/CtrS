"""90 Windows controller. Modes (chain90.ps1 runs them in this order and stops at the first nonzero exit):

  prepare  : QB, GF2, WV3 caches rebuilt from the H5 and checked against the 63 / 65 stage 3 hashes (as 88 / 89)
  optimize : on the idle GPU: structure checks (variants add no state, K2 targets unchanged, the resampling operators match the data
             operators), controls (C2MS-SP and 88 K2D, with and without the projection, on RR / FR test and the WV2 transfer), benchmark,
             MACs, sealed config, 58 preflight
  run      : two jobs at once. For every sensor and variant: <v>T (120 epochs) -> factorize -> <v>K (30 epochs) -> evaluation of <v>K
             (report only). The N variant (= the 89 confirmation model, lambda 1.2, c 0.25) of the three sensors goes first, then A, B, C.
             After every run the variant selection on validation is written (selection.json) and the controller ends.
             Stop rule (as 89, without the PSNR margin): on a process error, an incomplete run or a validation sanity failure (best epoch 0
             or nonfinite) no new job is launched, running jobs finish normally (never killed), stopped.json is written and the exit code is 3.
  factorize: K2 initial weights of <v>K from the hash-checked <v>T epoch-120 snapshot (83 factorization, 83 calibration patches)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess,platform,math
from pathlib import Path
import numpy as np
import h5py
from train import HERE,KEYS,sha,write,lock,now,KST,run_name,epochs,set_impl
from prepare import SENSORS,data_files
import model as M

PERF=HERE.parent;ROOT=PERF.parents[1]
SEED=42


def load_conf():return json.loads((HERE/'config.json').read_text(encoding='utf8'))


def source_hashes():
    e43=PERF/'43_normalized_parallel_capacity';e54=PERF/'54_multi_sensor_scale_mixed_training';e88=PERF/'88_factorization_at_decay_start';e89=PERF/'89_spectral_coarse_loss_weights'
    paths=[HERE/'model.py',HERE/'prepare.py',HERE/'train.py',HERE/'run.py',HERE/'config.json',HERE/'evaluate.py',e89/'model.py',e89/'prepare.py',e88/'model.py',e88/'prepare.py',
           PERF/'86_fft_adjoint_fast_inference/prepare.py',PERF/'85_hidden_width_reduction_fast_projection/prepare.py',PERF/'83_factorized_c2ms_finetuning/model.py',PERF/'83_factorized_c2ms_finetuning/prepare.py',PERF/'79_deep_narrow_branch_projection/model.py',PERF/'79_deep_narrow_branch_projection/prepare.py',
           PERF/'75_parallel_branch_width_depth/model.py',PERF/'58_spectral_loss_multi_sensor/train.py',PERF/'82_cross_sensor_band_calibration/prepare.py',
           e54/'model.py',e54/'prepare.py',e54/'train.py',e54/'evaluate.py',e54/'config.json',e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),
           ROOT/'common/interpolation.py',ROOT/'common/metrics.py',ROOT/'common/observation.py',ROOT/'common/evaluation_protocol.py']
    conf=load_conf()
    paths+=[ROOT/p for p in conf['cache_reference_manifest'].values()]
    paths+=[ROOT/conf['controls']['c2ms_sp'][s].replace('<seed>',str(SEED)) for s in conf['sensors']]
    paths+=[PERF/'82_cross_sensor_band_calibration/results/reference_stats/WV3.json']
    paths=list(dict.fromkeys(paths))
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def fingerprint(cfg):
    c={k:v for k,v in cfg.items() if k!='fingerprint'}
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


def runtime_cfg():
    conf=load_conf();root=Path(conf['runtime']);cp=root/'runtime_config.json'
    return root,cp,json.loads(cp.read_text(encoding='utf8'))


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
    """Variants add no state and keep the K2 targets; N equals the 89 2x2 model; the operators match the data operators."""
    import torch
    from common.interpolation import interp23tap
    from prepare import CoarsePairs
    out={};sh=lambda m:hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in sorted(m.state_dict().items()))).hexdigest()
    for sensor in ('QB','WV3'):
        bands=SENSORS[sensor]['bands'];t={a:sh(M.Model(a,bands,42,sensor=sensor)) for a in M.ARMS}
        if len({t[a] for a in M.FULL_SIZE})!=1 or len({t[a] for a in M.FACTORIZED})!=1:raise AssertionError('variants change the state '+sensor)
        if t['NT']!=sh(M.M89.Model('W2T',bands,42)) or t['NK']!=sh(M.M89.Model('W2K',bands,42)):raise AssertionError('N differs from the 89 2x2 model')
    n_t=[x[0] for x in M.M83.targets(M.Model('NT',4,42))]
    for a in M.FULL_SIZE:
        if [x[0] for x in M.M83.targets(M.Model(a,4,42,sensor='QB'))]!=n_t:raise AssertionError('factorization targets differ '+a)
    out['k2_targets']=len(n_t)
    P=M.P54;g=torch.Generator().manual_seed(3);pan=torch.rand(1,1,128,128,generator=g,dtype=torch.float64);ms=torch.rand(1,4,16,16,generator=g)
    out['pan_mtf_x4_vs_wald']={s:float((M.MTFDown(s,4,2)(pan.float()).double()-P.Wald(s,'cpu').pan(pan)).abs().max()) for s in ('QB','GF2','WV3')}
    out['ms_mtf_x2_kernel_vs_coarse_pairs']={s:float((M.MTFDown(s,2,0).km.double()-CoarsePairs(s,'cpu').ms_kernel.double()).abs().max()) for s in ('QB','GF2','WV3')}
    out['up23_two_stages_vs_interp23_x4']=float((M.Up23(0)(M.Up23(1)(ms))-interp23tap(ms,4)).abs().max())
    out['up23_first_stage_vs_interp23_x2']=float((M.Up23(1)(ms)-interp23tap(ms,2)).abs().max())
    if max(out['pan_mtf_x4_vs_wald'].values())>1e-4 or max(out['ms_mtf_x2_kernel_vs_coarse_pairs'].values())>1e-6 or out['up23_two_stages_vs_interp23_x4']>1e-5 or out['up23_first_stage_vs_interp23_x2']>1e-5:
        raise AssertionError('resampling operators differ from the data operators: '+json.dumps(out))
    up=interp23tap(ms,4)  # grid check: decimating the x4 interpolation at the data phases returns the MS samples best
    d=lambda a,b:float((M.MTFDown('QB',2,b)(M.MTFDown('QB',2,a)(up))-ms).abs().mean())
    out['grid_phase_0_then_1']=d(0,1);out['grid_phase_1_then_1']=d(1,1)
    if out['grid_phase_0_then_1']>=out['grid_phase_1_then_1']:raise AssertionError('internal decimation grid does not match the data grid')
    return out


def factorize(sensor,arm,seed):
    import torch
    import train as TR
    root,cp,cfg=runtime_cfg();set_impl(cfg);M.SENSOR['value']=sensor;dest=root/'init';dest.mkdir(exist_ok=True);p=TR.init_path(cfg,sensor,arm,seed);rec=p.with_suffix('.json')
    if rec.exists():print('FACTORIZED ALREADY',p.stem,flush=True);return
    trunk=M.SOURCE[arm];k,snap,meta=TR.snapshot_files(cfg,sensor,trunk,seed);src=snap/'latest.pt';ck=torch.load(src,map_location='cpu',weights_only=False)
    if ck['epoch']!=k or ck['fingerprint']!=cfg['fingerprint'] or ck['seed']!=seed:raise RuntimeError('snapshot is not the epoch-%d state'%k)
    source=dict(run=run_name(sensor,trunk,seed),file='snapshots/e%d/latest.pt'%k,epoch=k,sha256=meta['files']['latest.pt']['sha256'])
    TR.T.setup(seed)
    c={k_:np.load(root/'cache'/sensor/f'train_{k_}.npy',mmap_mode='r') for k_ in ('pan','lms','ms')}
    idx=np.sort(np.random.default_rng(M.CAL_SEED).choice(len(c['pan']),M.CAL_PATCHES,replace=False))
    cal={k_:torch.from_numpy(np.array(c[k_][idx])) for k_ in c};batches=[tuple(cal[k_][i:i+32] for k_ in ('pan','lms','ms')) for i in range(0,M.CAL_PATCHES,32)]
    base=M.Model(trunk,SENSORS[sensor]['bands'],seed);base.load_state_dict(ck['model'],strict=True)
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
    E.setup(SEED);set_impl(cfg);cfg['structure_checks']=structure_checks()
    for sensor in cfg['sensors']:
        for target in [sensor]+E.TRANSFER.get(sensor,[]):E.control_rows(cfg,sensor,target,SEED)
    cfg['controls_sha256']={p.name:sha(p) for p in sorted((root/'controls').glob('*.json'))}
    rows={}
    for sensor,arm in (('QB','NT'),('QB','CT'),('QB','NK'),('QB','CK'),('GF2','CT'),('WV3','CT')):
        log=(out/f'benchmark_{sensor}_{arm}.log').open('w',encoding='utf8')
        cfg_tmp=dict(cfg,fingerprint='benchmark');write(out/'bench_config.json',cfg_tmp)
        code=subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(out/'bench_config.json'),'--mode','bench','--sensor',sensor,'--arm',arm,'--layout',cfg['layout'],'--tag','n1'],stdout=log,stderr=subprocess.STDOUT).returncode;log.close()
        if code:raise RuntimeError('benchmark failed '+sensor+' '+arm)
        b=json.loads((out/f'benchmark_n1_{sensor}_{arm}.json').read_text(encoding='utf8'));rows[f'{sensor}_{arm}']=dict(median_step_seconds=b['median_step_seconds'],steps_per_epoch=b['steps_per_epoch'],peak_reserved_gib=b['peak_reserved_bytes']/2**30)
    cfg['benchmark']=rows
    shapes=lambda nb:[(1,1,256,256),(1,nb,256,256),(1,nb,64,64)]
    cfg['gmac_256']={s:{a:E.macs(M.Model(a,SENSORS[s]['bands'],42,sensor=s),shapes(SENSORS[s]['bands']))/1e9 for a in ('NT','NK','CK')} for s in cfg['sensors']}
    cfg['source_hashes']=source_hashes();cfg['fingerprint']=fingerprint(cfg);write(cp,cfg)
    subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode','preflight'],check=True)
    write(out/'optimization.json',dict(time_kst=now(),structure_checks=cfg['structure_checks'],benchmark=rows,gmac_256=cfg['gmac_256'],fingerprint=cfg['fingerprint']))
    print('OPTIMIZATION AND PREFLIGHT PASS',json.dumps(rows),flush=True)


def validation_of(root,key):
    d=json.loads((root/'runs'/run_name(*key)/'training_complete.json').read_text(encoding='utf8'))
    return dict(psnr=d['best_val']['val_psnr'],sam=d['best_val']['val_sam'],mse=d['best_val']['val_mse'],epoch=d['best_epoch'])


def sanity(cfg,root,key):
    v=validation_of(root,key);problems=[]
    if not all(math.isfinite(x) for x in (v['psnr'],v['sam'],v['mse'])):problems.append('nonfinite validation')
    if v['epoch']<1:problems.append('best epoch is 0 (no improvement over the initial state)')
    return problems


def select_variant(cfg,root):
    """config selection_rule on validation (the factorized final models): mean over QB, GF2, WV3 of the validation PSNR difference to N."""
    rule=cfg['selection_rule'];tau=rule['tau_sam_deg'];sensors=cfg['sensors']
    val={v:{s:validation_of(root,(s,v+'K',SEED)) for s in sensors} for v in M.VARIANTS}
    d={v:{s:dict(psnr=val[v][s]['psnr']-val['N'][s]['psnr'],sam=val[v][s]['sam']-val['N'][s]['sam']) for s in sensors} for v in M.VARIANTS if v!='N'}
    mean={v:sum(d[v][s]['psnr'] for s in sensors)/len(sensors) for v in d}
    order=sorted(d,key=lambda v:(-mean[v],M.VARIANTS.index(v)));checks=[];chosen='N'
    for v in order:
        bad={s:d[v][s]['sam'] for s in sensors if d[v][s]['sam']>tau[s]}
        checks.append(dict(variant=v,mean_delta_psnr=mean[v],rejected_by_sam=bad,positive=mean[v]>0))
        if mean[v]>0 and not bad:chosen=v;break
    return dict(time_kst=now(),rule=rule,validation=val,delta_vs_N=d,mean_delta_psnr=mean,checks=checks,chosen=chosen,
                note='validation only; test RR is reported in each evaluation and never used here')


def summary(cfg,root):
    out=dict(time_kst=now(),seed=SEED,runs={})
    for s in cfg['sensors']:
        for v in M.VARIANTS:
            e=root/'runs'/run_name(s,v+'K',SEED)/'evaluation_complete.json'
            if e.exists():out['runs'][f'{s}_{v}K']=json.loads(e.read_text(encoding='utf8'))['targets']
    for s in cfg['sensors']:
        for t in [s]+(['WV2'] if s=='WV3' else []):
            n=out['runs'].get(f'{s}_NK',{}).get(t)
            if not n:continue
            for v in ('A','B','C'):
                x=out['runs'].get(f'{s}_{v}K',{}).get(t)
                if x:out.setdefault('variant_vs_N_test',{}).setdefault(f'{s}->{t}',{})[v]={k:x['means'][v+'K + P']['RR'][k]-n['means']['NK + P']['RR'][k] for k in ('PSNR','SAM','ERGAS')}
    return out


def run(resume=False):
    conf=load_conf();root=Path(conf['runtime']);guard=lock(root/'controller.lock');cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));set_impl(cfg)
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes()!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    if not (root/'preflight/checks.json').exists():raise RuntimeError('preflight missing')
    if sha(root/'data_manifest.json')!=cfg['cache_manifest_sha256']:raise RuntimeError('cache manifest changed')
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint'],resume=resume))
    keys=[(s,v+a,SEED) for v in M.VARIANTS for s in ('GF2','QB','WV3') for a in ('T','K')]
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
    ready=lambda k:k[1] in M.FULL_SIZE or done((k[0],M.SOURCE[k[1]],k[2]),'train')
    jobs=[];running={}
    for k in keys:
        if not done(k,'train'):jobs.append(('train',k))
        elif k[1] in M.FACTORIZED and not done(k,'eval'):
            prob=sanity(cfg,root,k)
            if prob:stop(f'{run_name(*k)}: '+'; '.join(prob))
            jobs.append(('evaluate',k))
    rank=lambda j:(j[0]!='evaluate',j[1][1] not in M.FACTORIZED,keys.index(j[1]))  # evaluations, then ready factorized runs, then trunks in order
    while True:
        if state['stop'] is None:
            for j in sorted(jobs,key=rank):
                if len(running)>=cfg['max_parallel']:break
                kind,k=j
                if kind=='train' and not ready(k):continue
                if kind=='train' and k[1] in M.FACTORIZED:
                    if factorize_job(k):stop('factorize failed '+run_name(*k));break
                jobs.remove(j);running[j]=launch(kind,k)
        write(root/'queue_status.json',dict(time_kst=now(),running=[dict(job=j,sensor=k[0],arm=k[1]) for (j,k) in running],stop_reason=state['stop'],
              trained=[run_name(*k) for k in keys if done(k,'train')],evaluated=[run_name(*k) for k in keys if k[1] in M.FACTORIZED and done(k,'eval')]))
        time.sleep(5)
        for j,(p,log) in list(running.items()):
            if p.poll() is None:continue
            log.close();del running[j];kind,k=j
            if p.returncode or (kind=='train' and not done(k,'train')) or (kind=='evaluate' and not done(k,'eval')):
                stop(f'{kind} failed {run_name(*k)} (exit {p.returncode})');continue
            if kind=='train':
                prob=sanity(cfg,root,k)
                if prob:stop(f'{run_name(*k)}: '+'; '.join(prob));continue
                if k[1] in M.FACTORIZED:jobs.append(('evaluate',k))
        if not running and (state['stop'] is not None or not jobs):break
        if not running and jobs and all(j[0]=='train' and not ready(j[1]) for j in jobs):stop('no runnable job: '+str(jobs));break
    if state['stop']:
        write(root/'stopped.json',dict(time_kst=now(),reason=state['stop'],note='no new job was launched after the problem; running jobs finished normally'))
        write(root/'summary.json',summary(cfg,root));print('STOPPED',state['stop'],flush=True);sys.exit(3)
    sel=select_variant(cfg,root);write(root/'selection.json',sel);write(root/'summary.json',summary(cfg,root))
    write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],chosen=sel['chosen'],runs=len(keys),next='final model 3 seeds after the user decision'))
    print('ALL COMPLETE chosen',sel['chosen'],flush=True)


def screen():
    """Variant choice on 50-epoch screening runs (user decision 2026-11-21 02h, as the 89 lambda screening) instead of the 150-epoch
    A/B/C runs. Runs from a separate copy of the source tree on the same runtime after the sealed controller stopped at its first A/B/C
    trunk (blocked by provenance/block_abc.py; the stop must be exactly that lock failure). While the controller is still finishing its
    running N job, this uses only the free slot (queue_status.json); after the controller exits it takes controller.lock and both slots.
    Jobs: AS, BS, CS on GF2, QB, WV3 (validation only) and the N jobs the controller left (factorize + NK + evaluation).
    The N screening control is the 89 stage-B run BL1.20C0.25 (same recipe: full-size 2x2, 50 epochs, lambda 1.2, c 0.25, seed 42)."""
    conf=load_conf();root=Path(conf['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'));set_impl(cfg)
    note=lambda **k:print(json.dumps(dict(time_kst=now(),**k),ensure_ascii=False),flush=True)
    def stop_reason():  # the controller writes its stop reason to queue_status.json at once (stopped.json only when it exits)
        for f in ('stopped.json','queue_status.json'):
            try:
                r=json.loads((root/f).read_text(encoding='utf8')).get('reason' if f=='stopped.json' else 'stop_reason')
                if r:return r
            except Exception:pass
        return None
    while stop_reason() is None:
        if (root/'all_complete.json').exists():raise RuntimeError('controller completed; nothing to screen')
        time.sleep(30)
    reason=stop_reason()
    if not any(reason==f'train failed {s}_{v}T_s{SEED} (exit 1)' for s in cfg['sensors'] for v in 'ABC'):
        write(root/'screen_stopped.json',dict(time_kst=now(),reason='controller stop is not the intended A/B/C lock failure: '+repr(reason)));note(refused=reason);sys.exit(4)
    write(root/'screen_started.json',dict(time_kst=now(),controller_stop=reason,decision='user 2026-11-21 02h: choose the variant on 50-epoch screening runs',
          script=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__).resolve())))
    out=lambda k:root/'runs'/run_name(*k)
    trained=lambda k:(out(k)/'training_complete.json').exists();evaluated=lambda k:(out(k)/'evaluation_complete.json').exists()
    guard=[None]
    def controller_alive():
        if guard[0] is not None:return False
        f=(root/'controller.lock').open('a+b')
        try:
            import msvcrt;f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            f.close();return True
        guard[0]=f;note(controller='exited, lock taken');return False
    def controller_running():
        try:return len(json.loads((root/'queue_status.json').read_text(encoding='utf8'))['running'])
        except Exception:return cfg['max_parallel']
    screens=[(s,v+'S',SEED) for v in 'ABC' for s in ('GF2','QB','WV3')]
    n_jobs=lambda:[('train',(s,'NK',SEED)) for s in ('GF2','QB','WV3') if trained((s,'NT',SEED)) and not trained((s,'NK',SEED))]+\
                  [('evaluate',(s,'NK',SEED)) for s in ('GF2','QB','WV3') if trained((s,'NK',SEED)) and not evaluated((s,'NK',SEED))]
    jobs=[('train',k) for k in screens if not trained(k)];running={};problem=None;queued_n=set()
    def launch(kind,k):
        o=out(k);o.mkdir(parents=True,exist_ok=True);f=(o/(kind+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(HERE/'train.py'),'--config',str(cp),'--mode',kind,'--sensor',k[0],'--arm',k[1],'--seed',str(k[2])]
        if kind=='train' and (o/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT);note(launched=kind,run=run_name(*k),pid=p.pid);return p,f
    while True:
        alive=controller_alive()
        if not alive:  # N jobs only after the controller exited (it may still be training WV3 NT)
            for j in n_jobs():
                if j not in queued_n:queued_n.add(j);jobs.append(j)
        if problem is None:
            free=cfg['max_parallel']-len(running)-(controller_running() if alive else 0)
            for j in sorted(jobs,key=lambda j:(j[0]!='evaluate',j[1][1]!='NK',screens.index(j[1]) if j[1] in screens else 0)):
                if free<=0:break
                kind,k=j
                if kind=='train' and k[1]=='NK':
                    code=subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'--mode','factorize','--sensor',k[0],'--arm',k[1],'--seed',str(k[2])],
                                        stdout=(root/'init'/f'factorize_{run_name(*k)}.log').open('a',encoding='utf8'),stderr=subprocess.STDOUT).returncode
                    if code:problem='factorize failed '+run_name(*k);break
                jobs.remove(j);running[j]=launch(kind,k);free-=1
        time.sleep(10)
        for j,(p,f) in list(running.items()):
            if p.poll() is None:continue
            f.close();del running[j];kind,k=j
            ok=p.returncode==0 and (trained(k) if kind=='train' else evaluated(k))
            if ok and kind=='train' and sanity(cfg,root,k):ok=False
            note(finished=kind,run=run_name(*k),ok=ok,exit=p.returncode)
            if not ok and problem is None:problem=f'{kind} failed {run_name(*k)} (exit {p.returncode})'
        pending_n=bool(n_jobs()) and not alive
        if not running and not alive and (problem or (not jobs and not pending_n)):break
    if problem:
        write(root/'screen_stopped.json',dict(time_kst=now(),reason=problem,note='no new job after the problem; running jobs finished'));write(root/'summary.json',summary(cfg,root));note(stopped=problem);sys.exit(3)
    rule=cfg['selection_rule'];tau=rule['tau_sam_deg'];sensors=cfg['sensors']
    ref89=lambda s:json.loads((Path(conf['screening']['n_control_runtime'])/'runs'/f'{s}_BL1.20C0.25_s{SEED}'/'training_complete.json').read_text(encoding='utf8'))
    val={'N':{s:dict(psnr=ref89(s)['best_val']['val_psnr'],sam=ref89(s)['best_val']['val_sam'],epoch=ref89(s)['best_epoch']) for s in sensors}}
    val.update({v:{s:validation_of(root,(s,v+'S',SEED)) for s in sensors} for v in 'ABC'})
    d={v:{s:dict(psnr=val[v][s]['psnr']-val['N'][s]['psnr'],sam=val[v][s]['sam']-val['N'][s]['sam']) for s in sensors} for v in 'ABC'}
    mean={v:sum(d[v][s]['psnr'] for s in sensors)/len(sensors) for v in d}
    checks=[];chosen='N'
    for v in sorted(d,key=lambda v:(-mean[v],'ABC'.index(v))):
        bad={s:d[v][s]['sam'] for s in sensors if d[v][s]['sam']>tau[s]}
        checks.append(dict(variant=v,mean_delta_psnr=mean[v],rejected_by_sam=bad,positive=mean[v]>0))
        if mean[v]>0 and not bad:chosen=v;break
    sel=dict(time_kst=now(),rule=rule,screening=conf['screening'],validation=val,delta_vs_N=d,mean_delta_psnr=mean,checks=checks,chosen=chosen,
             note='50-epoch screening, validation only; the N control is the 89 run BL1.20C0.25 (same recipe, separate run: rerun spread up to 0.03 dB)')
    write(root/'screen_selection.json',sel);write(root/'summary.json',summary(cfg,root))
    write(root/'screen_complete.json',dict(time_kst=now(),chosen=chosen,next='final model: the chosen variant with the K2D 150-epoch recipe, seeds 42-44 (N: seed 42 exists), after the user decision'))
    note(chosen=chosen)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='90 controller: prepare -> optimize -> run (resume: --mode run --resume)')
    p.add_argument('--mode',choices=['prepare','optimize','run','factorize','screen'],required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--sensor');p.add_argument('--arm');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.mode=='factorize':factorize(a.sensor,a.arm,a.seed)
    else:{'prepare':prepare,'optimize':optimize,'screen':screen}.get(a.mode,lambda:run(a.resume))()
