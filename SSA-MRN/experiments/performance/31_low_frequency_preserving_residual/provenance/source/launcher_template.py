# Kaggle: paste this ENTIRE file into ONE code cell.
# P0 = retained G5 fine-tune; P1 = same G5 + high-pass residual.
MODE = 'train'                 # 'check' or 'train'
RUN_ID = '31_low_frequency_preserving_residual_s46_v1'
RESUME_FROM = ''                # exact prior run folder, /kaggle/working/... or attached Saved Output
PARENT_CHECKPOINT = ''          # optional exact path; otherwise hash-match input or pinned GitHub download
ALLOW_PARENT_DOWNLOAD = True    # set False for Internet OFF and attach the verified G5 best.pt
DATA_ROOT = '/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset'

import os,sys,json,time,hashlib,subprocess,shutil,zipfile,platform,urllib.request
from pathlib import Path
# _BUNDLE is embedded below by the build process; there is no git clone or pip install.


def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def _write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)

def _parent(work):
    expected='5a516b87c6e261c4251bfbebe38dc553c89e5720abc40c1e4e977eee6f5f5388'
    if PARENT_CHECKPOINT:
        p=Path(PARENT_CHECKPOINT)
        if not p.is_file() or _sha(p)!=expected:raise ValueError('PARENT_CHECKPOINT is not the verified experiment28 G5 seed46 best.pt')
        return p
    candidates=[]
    for p in Path('/kaggle/input').rglob('*.pt'):
        if p.stat().st_size==5968868:candidates.append(p)
    for p in candidates:
        if _sha(p)==expected:return p
    p=work/'parent_G5_best.pt'
    if p.exists():
        if _sha(p)!=expected:raise ValueError('Existing parent hash mismatch')
        return p
    if ALLOW_PARENT_DOWNLOAD:
        base='https://raw.githubusercontent.com/BIYONGHIYON/CtrS/60de05ef9f05f9f29ab5a50e59be391c7cfb9456/'
        url=base+'SSA-MRN/research/performance/28_residual_refinement/checkpoints/G5_QB_s46/best.pt'
        tmp=p.with_suffix('.download')
        print('Fetching pinned G5 parent; no other checkpoint is accepted.',flush=True)
        try:
            with urllib.request.urlopen(url,timeout=60) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst)
            if _sha(tmp)!=expected:
                raw=tmp.read_bytes()
                if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
                    media=url.replace('https://raw.githubusercontent.com/','https://media.githubusercontent.com/media/')
                    with urllib.request.urlopen(media,timeout=60) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst)
                if _sha(tmp)!=expected:raise ValueError('Parent download does not match verified SHA256')
            os.replace(tmp,p);return p
        except Exception as e:
            raise RuntimeError('G5 parent unavailable. Attach experiment28/checkpoints/G5_QB_s46/best.pt as Kaggle Input, set PARENT_CHECKPOINT, and rerun. Expected SHA256='+expected+'; download error='+str(e)) from e
    raise FileNotFoundError('Internet OFF: attach verified experiment28 G5 seed46 best.pt and set PARENT_CHECKPOINT.')

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
            if split in ('RR','FR') and n!=20:raise ValueError(f'{split}: expected 20 scenes, got {n}')
        manifests[split]=dict(path=str(p),sha256=_sha(p),keys=rows)
    if len({v['sha256'] for v in manifests.values()})!=4:raise ValueError('Duplicate split files')
    return manifests

def _plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist={a:json.loads((out/a/'history.json').read_text()) for a in ('P0','P1')}
    fig,axs=plt.subplots(1,4,figsize=(16,4))
    for a,rows in hist.items():
        for ax,key in zip(axs,('train_mse','val_mse','val_psnr','val_sam')):
            rr=[r for r in rows if r.get(key) is not None];ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=a);ax.set_title(key);ax.grid(alpha=.2)
    axs[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    metrics={a:json.loads((out/a/'results/metrics.json').read_text()) for a in ('P0','P1')}
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    deltas={}
    for ax,(proto,key) in zip(axs,[('RR','PSNR'),('RR','SAM'),('FR','QNR')]):
        v=[metrics[a]['protocols'][proto]['mean'][key] for a in ('P0','P1')];ax.bar(['P0','P1'],v);ax.set_title(proto+' '+key+(' (provisional)' if proto=='FR' else ''));deltas[proto+'_'+key]=v[1]-v[0]
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    paired={}
    for proto in ('RR','FR'):
        r0=metrics['P0']['protocols'][proto]['rows'];r1=metrics['P1']['protocols'][proto]['rows']
        if len(r0)!=20 or len(r1)!=20:raise ValueError('Incomplete test')
        paired[proto]=[dict(index=x['index'],delta={k:y['metrics'][k]-x['metrics'][k] for k in x['metrics']}) for x,y in zip(r0,r1) if x['index']==y['index']]
        if len(paired[proto])!=20:raise ValueError('Scene alignment mismatch')
    _write(out/'paired_scene_deltas.json',paired)
    # Thresholds are preregistered screening targets, not statistical significance.
    v0=min(r['val_mse'] for r in hist['P0']);v1=min(r['val_mse'] for r in hist['P1'])
    d0=metrics['P0']['protocols']['FR']['mean'];d1=metrics['P1']['protocols']['FR']['mean']
    _write(out/'summary.json',dict(deltas=deltas,validation_screen_pass=v1<=v0*1.005,rr_screen_pass=deltas['RR_PSNR']>=-.03 and deltas['RR_SAM']<=.02,fr_screen_pass=deltas['FR_QNR']>=.003 and d1['D_lambda']<=d0['D_lambda'] and d1['D_s']<=d0['D_s'],status='single-seed exploratory comparison; no automatic baseline promotion',metrics=metrics))

def _archive(out,complete):
    items={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ('manifest.json','controller.lock') and p.suffix not in ('.tmp','.download') and p.name!='run.lock':items[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=_sha(p))
    _write(out/'manifest.json',dict(files=items,state='complete' if complete else 'partial',remote_retention_verified=False))
    zip_path=out.parent/(out.name+('_results.zip' if complete else '_recovery.zip'))
    with zipfile.ZipFile(zip_path.with_suffix('.tmp'), 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*items,'manifest.json']:z.write(out/rel,arcname=out.name+'/'+rel)
    os.replace(zip_path.with_suffix('.tmp'),zip_path)
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
    zip_path.with_suffix('.zip.sha256').write_text(_sha(zip_path)+'  '+zip_path.name+'\n')
    print(('COMPLETE LOCAL OUTPUT: ' if complete else 'PARTIAL RECOVERY OUTPUT: ')+str(zip_path),flush=True)
    print('This confirms local output only. Verify Saved Version Output and download its ZIP before ending storage.',flush=True)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(zip_path)));display(FileLink(str(zip_path.with_suffix('.zip.sha256'))))
    except ImportError:pass

def main():
    import torch
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
        p.write_text(content,encoding='utf8')
    parent=_parent(out);retained=out/'parent_G5_best.pt'
    if parent.resolve()!=retained.resolve():shutil.copy2(parent,retained)
    data={k:str(Path(DATA_ROOT)/v) for k,v in dict(train='QuickBird__Training_Dataset__train_qb.h5',val='QuickBird__Training_Dataset__valid_qb.h5',RR='QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5',FR='QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5').items()}
    manifests=_inspect(data);_write(out/'data_manifest.json',manifests)
    cfg=dict(out=str(out),parent=str(retained),parent_sha256=_sha(retained),calibration=_CALIBRATION,data=data,seed=46,epochs=20,batch=64,micro=32,lr=1e-5,mode=MODE,resume=resume,gpu_map={'P0':0,'P1':1},scenes=[1,8,11,13,19],source_commit='60de05ef9f05f9f29ab5a50e59be391c7cfb9456',source_hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in _BUNDLE.items()},precision='AMP FP16 train; FP32 validation/test',selection='minimum validation MSE including epoch0; earliest exact tie',parent_condition='experiment28 G5 QB seed46, fresh Adam fine-tuning, not original optimizer resume')
    stable={k:v for k,v in cfg.items() if k not in ('out','parent','resume','data','mode')};stable['data_hashes']={k:v['sha256'] for k,v in manifests.items()}
    cfg['fingerprint']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    cp=out/'config.json'
    if resume and json.loads(cp.read_text())['fingerprint']!=cfg['fingerprint']:raise ValueError('Resume incompatible with source/config/data')
    _write(cp,cfg);_write(out/'selection.json',dict(seed=46,indices=cfg['scenes'],index_base=0,unit='full RR scene',rule='fixed before training/test scoring',display_order=cfg['scenes'],panels=['LR MS nearest','PAN','prediction','GT'],rgb_bands=[2,1,0],contrast='per-scene GT 1st/99th percentile shared across MS panels',FR_has_GT=False))
    handles=[];complete=False
    try:
        for arm,gpu in cfg['gpu_map'].items():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
            f=(out/arm/'worker.log');f.parent.mkdir(exist_ok=True);stream=f.open('a',encoding='utf8')
            proc=subprocess.Popen([sys.executable,'-u',str(src/'worker.py'),str(cp),arm],env=env,stdout=stream,stderr=subprocess.STDOUT);handles.append((arm,proc,stream))
        last={}
        while any(p.poll() is None for _,p,_ in handles):
            for arm,p,f in handles:
                status=out/arm/'status.json'
                if status.exists():
                    s=status.read_text()
                    if s!=last.get(arm):print(s,flush=True);last[arm]=s
            # Fail promptly and request a safe checkpoint from a surviving peer.
            if any(p.poll() is not None and p.returncode!=0 for _,p,_ in handles):
                for _,p,_ in handles:
                    if p.poll() is None:p.terminate()
                break
            time.sleep(5)
        for arm,p,f in handles:
            rc=p.wait();f.close()
            if rc:
                print((out/arm/'worker.log').read_text()[-12000:],flush=True)
                raise RuntimeError(f'{arm} failed, exit {rc}; see worker.log')
        if MODE=='check':
            print('CHECK PASS: two GPU-resident caches, strict parent loading, finite forward/backward; training not started.',flush=True);return
        for arm in ('P0','P1'):
            if not (out/arm/'complete.json').exists():raise RuntimeError(f'{arm} stopped safely before completion; resume required')
        init=[json.loads((out/a/'initialization.json').read_text()) for a in ('P0','P1')]
        if init[0]['state_sha256']!=init[1]['state_sha256']:raise ValueError('Initial states differ')
        hs=[json.loads((out/a/'history.json').read_text())[1:] for a in ('P0','P1')]
        if [r['order_sha256'] for r in hs[0]]!=[r['order_sha256'] for r in hs[1]]:raise ValueError('Actual epoch orders differ')
        for a in ('P0','P1'):
            for i in cfg['scenes']:
                if not (out/a/'results'/f'RR_scene_{i:03}.png').is_file():raise ValueError('Missing selected preview')
        _plots(out);complete=True
    finally:
        for _,p,f in handles:
            if p.poll() is None:p.terminate()
        for _,p,f in handles:
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            if not f.closed:f.close()
        _archive(out,complete)

main()
