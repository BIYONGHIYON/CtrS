# 32: MS-anchored reversible brightness/contrast normalization.
# Paste this ENTIRE file into ONE Kaggle cell. No Internet or prior weights needed.
# Choose GPU T4 x2; attach the exact dataset named below.
# MODE='train' includes GPU preflight, 100 epochs, RR/FR scoring and ZIP creation.
# MODE='check' runs GPU preflight only; use a NEW RUN_ID when starting real training.
# A rerun never silently overwrites an existing folder.
MODE = 'train'                   # 'check' or 'train'
SEED = 42                       # later independent repetitions: 43, 44
RUN_ID = '32_ms_anchored_reversible_normalization_s42_v1'
RESUME_FROM = ''                # exact prior RUN_ID folder with config.json
EVALUATE_AFTER_TRAINING = True   # False permits validation-only screening; resume with True to score
DATA_ROOT = '/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset'

import os,sys,json,time,hashlib,subprocess,shutil,zipfile,platform,urllib.request
from pathlib import Path
# EMBEDDED_SOURCE_BUNDLE


def _executed_source():
    # Snapshot the complete pasted cell (or script), not a mutable remote URL.
    try:
        path=globals().get('__file__')
        if path and Path(path).is_file():
            source=Path(path).read_text(encoding='utf8')
            if '_BUNDLE = ' in source:return source,'script_file'
    except (OSError,UnicodeError):pass
    try:
        shell=get_ipython()
        for source in reversed(shell.history_manager.input_hist_raw):
            if 'def _inspect(' in source and '_BUNDLE = ' in source and 'def main(' in source:
                return source,'notebook_cell'
    except (NameError,AttributeError):pass
    # Script exec() users may not provide source history. Reconstruct the unedited
    # delivered controller; runtime parameters are separately preserved in config.
    source=_BUNDLE['controller_template.py'].replace('# EMBEDDED_SOURCE_BUNDLE','_BUNDLE = '+repr(_BUNDLE),1)
    return source,'reconstructed_delivered_controller'

def _controller_fingerprint(source):
    import ast
    tree=ast.parse(source)
    allowed={'MODE','RUN_ID','RESUME_FROM','EVALUATE_AFTER_TRAINING','DATA_ROOT'}
    tree.body=[node for node in tree.body if not (isinstance(node,ast.Assign) and
               any(isinstance(target,ast.Name) and target.id in allowed for target in node.targets))]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def _write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)


def _inspect(data):
    import h5py,numpy as np
    manifests={}
    for split,path in data.items():
        p=Path(path)
        if not p.is_file():raise FileNotFoundError(f'{split}: missing exact dataset path: {p}')
        rows={}
        with h5py.File(p,'r') as f:
            required=('pan','lms','ms','gt') if split in ('train','val','RR') else ('pan','lms','ms')
            if any(k not in f for k in required):raise ValueError(f'{split} missing keys {required}')
            n=len(f['pan']);h,w=f['pan'].shape[-2:]
            if f['pan'].shape!=(n,1,h,w) or f['lms'].shape!=(n,4,h,w) or f['ms'].shape!=(n,4,h//4,w//4):raise ValueError(f'{split}: unexpected NCHW or spatial ratio')
            if 'gt' in required and f['gt'].shape!=(n,4,h,w):raise ValueError(f'{split}: GT shape')
            if min(h,w)<16 or h%4 or w%4:raise ValueError(f'{split}: invalid spatial dimensions')
            for k in required:
                ds=f[k];lo=float('inf');hi=-float('inf')
                for start in range(0,n,256):
                    a=np.asarray(ds[start:start+256]);
                    if not np.isfinite(a).all():raise ValueError(f'{split}/{k}: nonfinite values')
                    lo=min(lo,float(a.min()));hi=max(hi,float(a.max()))
                # Cubic LMS ringing and tiny negative MS roundoff are valid. Never clip or min-max normalize.
                if max(abs(lo),abs(hi))>32768:raise ValueError(f'{split}/{k}: suspicious DN range {lo},{hi}')
                if k in ('pan','gt') and hi<=2:raise ValueError(f'{split}/{k}: expected raw DN values, not normalized data')
                rows[k]=dict(shape=list(ds.shape),dtype=str(ds.dtype),min=lo,max=hi)
                print(f'{split}/{k}: {lo:.8g} to {hi:.8g}; /2047 preserved',flush=True)
            expected={'train':17139,'val':1905,'RR':20,'FR':20}[split]
            if n!=expected:raise ValueError(f'{split}: expected {expected} samples, got {n}')
            if split in ('RR','FR') and n!=20:raise ValueError(f'{split}: expected 20 scenes, got {n}')
        manifests[split]=dict(path=str(p),sha256=_sha(p),keys=rows)
    if len({v['sha256'] for v in manifests.values()})!=4:raise ValueError('Duplicate split files')
    return manifests

def _plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist={a:json.loads((out/'results'/a/'history.json').read_text()) for a in ('N0','N1')}
    fig,axs=plt.subplots(1,4,figsize=(16,4))
    for a,rows in hist.items():
        for ax,key in zip(axs,('train_mse','val_mse','val_psnr','val_sam')):
            rr=[r for r in rows if r.get(key) is not None];ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=a);ax.set_title(key);ax.grid(alpha=.2)
    axs[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    metrics={a:json.loads((out/'results'/a/'results/metrics.json').read_text()) for a in ('N0','N1')}
    fig,axs=plt.subplots(2,4,figsize=(16,8))
    deltas={}
    for ax,(proto,key) in zip(axs.flat,[('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('RR','SCC'),('RR','Q2n'),('FR','QNR'),('FR','D_lambda'),('FR','D_s')]):
        v=[metrics[a]['protocols'][proto]['mean'][key] for a in ('N0','N1')];ax.bar(['N0','N1'],v);ax.set_title(proto+' '+key+(' (provisional)' if proto=='FR' else ''));deltas[proto+'_'+key]=v[1]-v[0]
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    paired={}
    for proto in ('RR','FR'):
        r0=metrics['N0']['protocols'][proto]['rows'];r1=metrics['N1']['protocols'][proto]['rows']
        if len(r0)!=20 or len(r1)!=20:raise ValueError('Incomplete test')
        paired[proto]=[dict(index=x['index'],delta={k:y['metrics'][k]-x['metrics'][k] for k in x['metrics']}) for x,y in zip(r0,r1) if x['index']==y['index']]
        if len(paired[proto])!=20:raise ValueError('Scene alignment mismatch')
    _write(out/'paired_scene_deltas.json',paired)
    # Thresholds are preregistered screening targets, not statistical significance.
    v0=min(r['val_mse'] for r in hist['N0']);v1=min(r['val_mse'] for r in hist['N1'])
    d0=metrics['N0']['protocols']['FR']['mean'];d1=metrics['N1']['protocols']['FR']['mean']
    _write(out/'summary.json',dict(deltas=deltas,validation_screen_pass=v1<=v0*1.005,rr_screen_pass=deltas['RR_PSNR']>=-.03 and deltas['RR_SAM']<=.02,fr_screen_pass=deltas['FR_QNR']>=.003 and d1['D_lambda']<=d0['D_lambda'] and d1['D_s']<=d0['D_s'],status='single-seed exploratory comparison; no automatic baseline promotion',metrics=metrics))

def _verify_pair(out,cfg):
    arms=('N0','N1')
    initial=[json.loads((out/'results'/arm/'initialization.json').read_text()) for arm in arms]
    if initial[0]['state_sha256']!=initial[1]['state_sha256']:raise ValueError('Common initialization differs')
    histories=[json.loads((out/'results'/arm/'history.json').read_text()) for arm in arms]
    for history in histories:
        if [r['epoch'] for r in history]!=list(range(cfg['epochs']+1)):raise ValueError('Missing epoch records')
    orders=[[r['order_sha256'] for r in history[1:]] for history in histories]
    if orders[0]!=orders[1]:raise ValueError('Actual batch orders differ')
    skipped=[history[-1]['skipped'] for history in histories]
    if any(skipped):raise ValueError('AMP skipped updates found. Results preserved, comparison requires review: '+str(skipped))
    _write(out/'pair_verification.json',dict(initial_parameters_equal=True,actual_epoch_orders_equal=True,
            completed_epochs=cfg['epochs'],skipped_updates=skipped,bitwise_cuda_reproducibility_guaranteed=False))

def _training_plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(16,4))
    for arm in ('N0','N1'):
        rows=json.loads((out/'results'/arm/'history.json').read_text())
        for ax,key in zip(axs,('train_mse','val_mse','val_psnr','val_sam')):
            rr=[r for r in rows if r.get(key) is not None]
            ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm)
            ax.set_title(key);ax.grid(alpha=.2)
    axs[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)

def _verify_artifacts(out,cfg):
    needed=['curves.png','comparison.png','pair_verification.json','paired_scene_deltas.json','summary.json','selection.json','data_manifest.json','config.json']
    for arm in ('N0','N1'):
        needed += [f'checkpoints/{arm}_QB_s{cfg["seed"]}/{name}.pt' for name in ('best','latest')]
        needed += [f'results/{arm}/{name}' for name in ('preflight.json','initialization.json','history.json','complete.json','results/metrics.json')]
        needed += [f'results/{arm}/results/RR_scene_{i:03}.png' for i in cfg['scenes']]
    for path in needed:
        if not (out/path).is_file() or (out/path).stat().st_size==0:raise ValueError('Missing artifact: '+path)
    for arm in ('N0','N1'):
        rec=json.loads((out/'results'/arm/'complete.json').read_text())
        ck=out/'checkpoints'/f'{arm}_QB_s{cfg["seed"]}'
        for name in ('best','latest'):
            if rec[name+'_sha256']!=_sha(ck/(name+'.pt')):raise ValueError('Checkpoint hash mismatch')
    _write(out/'artifact_verification.json',dict(required_files=needed,all_exist_nonempty=True,checkpoint_hashes_match=True,
           training_complete=True,evaluation_complete=True,remote_retention_verified=False))


def _archive(out,complete):
    items={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ('manifest.json','controller.lock') and p.suffix not in ('.tmp','.download') and p.name!='run.lock':items[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=_sha(p))
    _write(out/'manifest.json',dict(files=items,state='complete' if complete else ('training_complete_evaluation_pending' if (out/'training_only.json').exists() else 'partial'),remote_retention_verified=False,checkpoint_resume_requires='both latest.pt and matching config/source/data'))
    zip_path=out.parent/(out.name+('_results.zip' if complete else ('_training.zip' if (out/'training_only.json').exists() else '_recovery.zip')))
    with zipfile.ZipFile(zip_path.with_suffix('.tmp'), 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*items,'manifest.json']:z.write(out/rel,arcname=out.name+'/'+rel)
    os.replace(zip_path.with_suffix('.tmp'),zip_path)
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        for rel,info in items.items():
            if hashlib.sha256(z.read(out.name+'/'+rel)).hexdigest()!=info['sha256']:raise ValueError('ZIP SHA256 mismatch: '+rel)
    zip_path.with_suffix('.zip.sha256').write_text(_sha(zip_path)+'  '+zip_path.name+'\n')
    print(('COMPLETE LOCAL OUTPUT: ' if complete else 'PARTIAL RECOVERY OUTPUT: ')+str(zip_path),flush=True)
    print('This confirms local output only. Verify Saved Version Output and download its ZIP before ending storage.',flush=True)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(zip_path)));display(FileLink(str(zip_path.with_suffix('.zip.sha256'))))
    except ImportError:pass

def main():
    import importlib.util
    missing=[name for name in ('torch','h5py','numpy','scipy','skimage','PIL','matplotlib') if importlib.util.find_spec(name) is None]
    if missing:raise RuntimeError('Missing dependencies: '+str(missing)+'; use a Kaggle GPU notebook image with these packages.')
    import torch
    if not RUN_ID.replace('_','').replace('-','').isalnum():raise ValueError('RUN_ID must be a plain folder name')
    if ('_s'+str(SEED)+'_') not in RUN_ID:raise ValueError('Update RUN_ID to match SEED')
    if MODE not in ('check','train'):raise ValueError('MODE must be check or train')
    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4 x2 before running; no CPU/single-GPU fallback.')
    if any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError('This fixed run targets two T4 GPUs.')
    print('GPUs:',[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
    out=Path('/kaggle/working')/RUN_ID
    import fcntl
    control_lock=Path('/kaggle/working')/(RUN_ID+'.controller.lock');lock=control_lock.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resume=bool(RESUME_FROM)
    if resume:
        old=Path(RESUME_FROM)
        if (old/'manifest.json').is_file():
            previous=json.loads((old/'manifest.json').read_text())
            for rel,info in previous['files'].items():
                path=old/rel
                if not path.is_file() or path.stat().st_size!=info['bytes'] or _sha(path)!=info['sha256']:raise ValueError('Saved Output integrity check failed: '+rel)
        if not (old/'config.json').exists():raise FileNotFoundError('RESUME_FROM must be the exact prior run folder containing config.json')
        if old.resolve()!=out.resolve():
            if out.exists():raise FileExistsError('Destination exists; choose exact same-session resume or a fresh RUN_ID')
            shutil.copytree(old,out)
    elif out.exists():raise FileExistsError(f'Existing run is protected: {out}. Set RESUME_FROM to resume or change RUN_ID for new work.')
    out.mkdir(parents=True,exist_ok=True)
    _write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpus=[torch.cuda.get_device_name(i) for i in range(2)]))
    src=out/'provenance/source'
    for name,content in _BUNDLE.items():
        p=src/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=content.encode('utf8'):raise ValueError('Saved source differs; start a new run')
        p.write_bytes(content.encode('utf8'))
    data={k:str(Path(DATA_ROOT)/v) for k,v in dict(train='QuickBird__Training_Dataset__train_qb.h5',val='QuickBird__Training_Dataset__valid_qb.h5',RR='QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5',FR='QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5').items()}
    manifests=_inspect(data);_write(out/'data_manifest.json',manifests)
    executed,source_capture=_executed_source()
    controller_hash=_controller_fingerprint(executed)
    cfg=dict(controller_sha256=controller_hash,out=str(out),data=data,seed=SEED,epochs=100,batch=64,micro=32,lr=1e-4,std_floor=0.01,
             mode=MODE,resume=resume,gpu_map={'N0':0,'N1':1},scenes=[1,8,11,13,19],
             evaluate_after_training=EVALUATE_AFTER_TRAINING,
             source_commit='2878e9bbf273edaeafd2b1251a8aed0227f06a19',
             source_hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in _BUNDLE.items()},
             precision='AMP FP16 train; FP32 stats/inverse/loss/validation/test; TF32 off',
             selection='minimum validation MSE including epoch0; earliest exact tie',
             experiment=32,research_track='performance',experimenter='김경찬',
             model='fresh original RestoredPansharpeningNet K4; N1 shared per-sample MS affine wrapper',
             data_normalization='DN/2047; no clipping; GT excluded from input statistics')
    stable={k:v for k,v in cfg.items() if k not in ('out','resume','data','mode','evaluate_after_training')};stable['data_hashes']={k:v['sha256'] for k,v in manifests.items()}
    cfg['fingerprint']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    cp=out/'config.json'
    if resume and json.loads(cp.read_text())['fingerprint']!=cfg['fingerprint']:raise ValueError('Resume incompatible with source/config/data')
    for arm in ('N0','N1'):
        folder=out/'results'/arm
        for name in ('failure.json','complete.json','training_complete.json'):
            marker=folder/name
            if marker.exists():marker.unlink()
    if (out/'training_only.json').exists():(out/'training_only.json').unlink()
    (out/'provenance'/'run_cell.py').write_text(executed,encoding='utf8')
    _write(out/'provenance'/'controller_capture.json',dict(method=source_capture,canonical_sha256=controller_hash,raw_sha256=hashlib.sha256(executed.encode()).hexdigest()))
    _write(cp,cfg);_write(out/'selection.json',dict(seed=SEED,indices=cfg['scenes'],index_base=0,unit='full RR scene',rule='fixed before training/test scoring',display_order=cfg['scenes'],panels=['LR MS nearest','PAN','prediction','GT'],rgb_bands=[2,1,0],contrast='per-scene GT 1st/99th percentile shared across MS panels',FR_has_GT=False))
    _write(out/'protocol.json',dict(experiment=32,RR='20 full scenes, peak2047 per-band PSNR averaged per scene',FR='20 full scenes; QNR/D_lambda/D_s provisional',std='population variance over C,H,W of LR MS per sample, floor=0.01',loss='MSE after inverse in DN/2047 units',predefined_targets=dict(RR_PSNR_min_delta=-.03,RR_SAM_max_delta=.02,FR_QNR_min_delta=.003),test_tuning=False))
    handles=[];complete=False
    try:
        for arm,gpu in cfg['gpu_map'].items():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
            f=(out/'results'/arm/'worker.log');f.parent.mkdir(parents=True,exist_ok=True);stream=f.open('a',encoding='utf8')
            proc=subprocess.Popen([sys.executable,'-u',str(src/'worker.py'),str(cp),arm],env=env,stdout=stream,stderr=subprocess.STDOUT);handles.append((arm,proc,stream))
        last={};initial_pair_checked=False
        while any(p.poll() is None for _,p,_ in handles):
            for arm,p,f in handles:
                status=out/'results'/arm/'status.json'
                if status.exists():
                    s=status.read_text()
                    if s!=last.get(arm):print(s,flush=True);last[arm]=s
            initial_files=[out/'results'/a/'initialization.json' for a in ('N0','N1')]
            if not initial_pair_checked and all(p.exists() for p in initial_files):
                initial_pair_checked=True
                if len({json.loads(p.read_text())['state_sha256'] for p in initial_files})!=1:raise ValueError('Common initialization differs; stopping both workers')
            # Fail promptly and request a safe checkpoint from a surviving peer.
            if any(p.poll() is not None and p.returncode!=0 for _,p,_ in handles):
                for _,p,_ in handles:
                    if p.poll() is None:p.terminate()
                break
            time.sleep(5)
        for arm,p,f in handles:
            rc=p.wait();f.close()
            if rc:
                print((out/'results'/arm/'worker.log').read_text()[-12000:],flush=True)
                raise RuntimeError(f'{arm} failed, exit {rc}; see results/{arm}/worker.log')
        if MODE=='check':
            print('CHECK PASS: two GPU-resident caches, common fresh initialization, finite forward/backward; training not started.',flush=True);return
        if not EVALUATE_AFTER_TRAINING:
            for arm in ('N0','N1'):
                if not (out/'results'/arm/'training_complete.json').is_file():raise RuntimeError('Training interrupted; resume required')
            _verify_pair(out,cfg)
            _training_plots(out)
            _write(out/'training_only.json',dict(state='training_complete_evaluation_pending',test_scored=False))
            print('TRAINING COMPLETE; RR/FR intentionally not scored. Inspect validation before further experiments.',flush=True)
            return
        for arm in ('N0','N1'):
            if not (out/'results'/arm/'complete.json').exists():raise RuntimeError(f'{arm} stopped safely before completion; resume required')
        init=[json.loads((out/'results'/a/'initialization.json').read_text()) for a in ('N0','N1')]
        if init[0]['state_sha256']!=init[1]['state_sha256']:raise ValueError('Initial states differ')
        hs=[json.loads((out/'results'/a/'history.json').read_text())[1:] for a in ('N0','N1')]
        if [r['order_sha256'] for r in hs[0]]!=[r['order_sha256'] for r in hs[1]]:raise ValueError('Actual epoch orders differ')
        for a in ('N0','N1'):
            for i in cfg['scenes']:
                if not (out/'results'/a/'results'/f'RR_scene_{i:03}.png').is_file():raise ValueError('Missing selected preview')
        _verify_pair(out,cfg);_plots(out);_verify_artifacts(out,cfg);complete=True
    finally:
        for _,p,f in handles:
            if p.poll() is None:p.terminate()
        for _,p,f in handles:
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            if not f.closed:f.close()
        _archive(out,complete)

if __name__ == '__main__':
    main()

