# Experiment40: shared-MS centering/scaling factorial ablation.
# Paste this ENTIRE delivered run.py into ONE Kaggle code cell.
MODE = 'train'                 # 'check' uses a separate RUN_ID and does not train
SEED = 42
RUN_ID = '40_ms_center_scale_ablation_s42_v1'
RESUME_FROM = ''               # exact previous run folder; keep code/seed/settings fixed
EVALUATE_AFTER_TRAINING = False # resume same run with True AFTER validation decision
DATA_ROOT = '/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset'

import os,sys,json,time,hashlib,subprocess,shutil,zipfile,platform,ast
from pathlib import Path
from datetime import datetime,timedelta,timezone
ARMS=('A0','A1','A2','A3')
WAVES=(('A0','A3'),('A1','A2'))
KST=timezone(timedelta(hours=9))
# EMBEDDED_SOURCE_BUNDLE

def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def _write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)

def _executed_source():
    path=globals().get('__file__')
    if path and Path(path).is_file():
        s=Path(path).read_text(encoding='utf8')
        if '_BUNDLE = ' in s and 'def _run_wave(' in s:return s,'script_file'
    try:
        for s in reversed(get_ipython().history_manager.input_hist_raw):
            if '_BUNDLE = ' in s and 'def _run_wave(' in s and 'def main(' in s:return s,'notebook_cell'
    except (NameError,AttributeError):pass
    return _BUNDLE['controller_template.py'].replace('# EMBEDDED_SOURCE_BUNDLE','_BUNDLE = '+repr(_BUNDLE),1),'reconstructed_template_runtime_config_separate'

def _controller_fingerprint(source):
    tree=ast.parse(source);allowed={'MODE','RUN_ID','RESUME_FROM','EVALUATE_AFTER_TRAINING','DATA_ROOT'}
    tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in allowed for t in n.targets))]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()

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


def _verify_training(out,cfg):
    hashes=[];orders=[];best={}
    import torch
    for arm in ARMS:
        r=out/'results'/arm;c=out/'checkpoints'/f'{arm}_QB_s{cfg["seed"]}'
        if not (r/'training_complete.json').is_file():raise RuntimeError(f'{arm}: incomplete training; resume required')
        hashes.append(json.loads((r/'initialization.json').read_text())['state_sha256'])
        h=json.loads((r/'history.json').read_text())
        if [x['epoch'] for x in h]!=list(range(cfg['epochs']+1)):raise ValueError('Missing epoch records: '+arm)
        orders.append([x['order_sha256'] for x in h[1:]])
        if h[-1]['skipped'] or h[-1]['updates']!=cfg['epochs']*((17139+cfg['batch']-1)//cfg['batch']):raise ValueError('AMP skip/update mismatch: '+arm)
        selected=min(h,key=lambda x:x['val_mse'])
        if not(c/'initial.pt').is_file():raise ValueError('Missing original initialization: '+arm)
        bc=torch.load(c/'best.pt',map_location='cpu',weights_only=False)
        lc=torch.load(c/'latest.pt',map_location='cpu',weights_only=False)
        if bc['best_epoch']!=selected['epoch'] or bc['metrics']['val_mse']!=selected['val_mse']:raise ValueError('Validation best mismatch: '+arm)
        if lc['epoch']!=cfg['epochs'] or lc['next_batch']!=0:raise ValueError('Latest cursor not at training completion')
        for k,v in bc['model'].items():
            if not torch.equal(v,lc['best_model'][k]):raise ValueError('latest best_model differs: '+arm)
        best[arm]=dict(epoch=selected['epoch'],val_mse=selected['val_mse'],best_sha256=_sha(c/'best.pt'))
    if len(set(hashes))!=1:raise ValueError('Four-arm initial state mismatch')
    if any(x!=orders[0] for x in orders[1:]):raise ValueError('Four-arm actual epoch order mismatch')
    passed={a:best[a]['val_mse']<=best['A3']['val_mse']*1.005 for a in ('A1','A2')}
    result=dict(best=best,simplified_arm_pass=passed,repeat_all_four_arms_seeds43_44=any(passed.values()),
                rule='A1 or A2 best validation MSE <= A3*1.005; frozen before test; no automatic seed launches',
                test_used_for_decision=False)
    _write(out/'validation_decision.json',result)
    _write(out/'pair_verification.json',dict(all_four_initial_states_equal=True,all_four_epoch_orders_equal=True,
                completed_epochs=cfg['epochs'],skipped_updates=0,validation_best_and_latest_backup_verified=True,
                cuda_bitwise_reproducibility_guaranteed=False))
    print('VALIDATION DECISION:',json.dumps(result,ensure_ascii=False),flush=True)
    return result

def _training_plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(16,4))
    for arm in ARMS:
        h=json.loads((out/'results'/arm/'history.json').read_text())
        for ax,key in zip(axs,('train_mse','val_mse','val_psnr','val_sam')):
            rows=[r for r in h if r.get(key) is not None];ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=arm);ax.set_title(key);ax.grid(alpha=.2)
    axs[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)

def _plots(out):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics={a:json.loads((out/'results'/a/'results/metrics.json').read_text()) for a in ARMS}
    fig,axs=plt.subplots(2,4,figsize=(16,8));deltas={};paired={}
    for ax,(proto,key) in zip(axs.flat,[('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('RR','SCC'),('RR','Q2n'),('FR','QNR'),('FR','D_lambda'),('FR','D_s')]):
        values=[metrics[a]['protocols'][proto]['mean'][key] for a in ARMS]
        ax.bar(ARMS,values);ax.set_title(proto+' '+key+(' (provisional)' if proto=='FR' or key in ('SCC','Q2n') else ''))
        deltas[proto+'_'+key]={a:values[i]-values[0] for i,a in enumerate(ARMS)}
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    for proto in ('RR','FR'):
        rows={a:metrics[a]['protocols'][proto]['rows'] for a in ARMS}
        for a in ARMS:
            if [r['index'] for r in rows[a]]!=list(range(20)):raise ValueError('Incomplete/unmatched scenes: '+a)
        paired[proto]=[{ 'index':i,'delta_from_A0':{a:{k:rows[a][i]['metrics'][k]-rows['A0'][i]['metrics'][k] for k in rows[a][i]['metrics']} for a in ARMS},
                         'delta_from_A3':{a:{k:rows[a][i]['metrics'][k]-rows['A3'][i]['metrics'][k] for k in rows[a][i]['metrics']} for a in ARMS}} for i in range(20)]
    tests={}
    for a in ('A1','A2','A3'):
        tests[a]=dict(rr_preservation=deltas['RR_PSNR'][a]>=-.03 and deltas['RR_SAM'][a]<=.02,
                      provisional_fr_improvement=deltas['FR_QNR'][a]>=.003 and deltas['FR_D_lambda'][a]<=0 and deltas['FR_D_s'][a]<=0)
    replacement={}
    for a in ('A1','A2'):
        d={key:values[a]-values['A3'] for key,values in deltas.items()}
        replacement[a]=dict(deltas_from_A3=d,rr_preservation=d['RR_PSNR']>=-.03 and d['RR_SAM']<=.02,provisional_fr_improvement=d['FR_QNR']>=.003 and d['FR_D_lambda']<=0 and d['FR_D_s']<=0)
    interaction={key:d['A3']-d['A1']-d['A2'] for key,d in deltas.items()}
    _write(out/'summary.json',dict(metrics=metrics,deltas_from_A0=deltas,screen_vs_A0=tests,replacement_screen_vs_A3=replacement,
          factorial_interaction_A3_minus_A1_minus_A2_plus_A0=interaction,
          limits='Single-seed exploratory effects, not a causal proof or significance test; QNR/Q2n/SCC provisional; no automatic adoption'))
    _write(out/'paired_scene_deltas.json',paired)

def _verify_artifacts(out,cfg):
    needed=['curves.png','comparison.png','summary.json','paired_scene_deltas.json','validation_decision.json','pair_verification.json']
    for a in ARMS:
        needed += [f'checkpoints/{a}_QB_s{cfg["seed"]}/{n}.pt' for n in ('initial','best','latest')]
        needed += [f'results/{a}/{n}' for n in ('complete.json','preflight.json','initialization.json','history.json','input_statistics.json','results/metrics.json','results/evaluated_weight.json','results/input_statistics.json','results/trained_affine_diagnostics.json')]
        needed += [f'results/{a}/results/RR_scene_{i:03}.png' for i in cfg['scenes']]
        rec=json.loads((out/'results'/a/'complete.json').read_text());c=out/'checkpoints'/f'{a}_QB_s{cfg["seed"]}'
        for n in ('best','latest'):
            if _sha(c/(n+'.pt'))!=rec[n+'_sha256']:raise ValueError('Weight hash mismatch')
    for n in needed:
        if not(out/n).is_file() or (out/n).stat().st_size==0:raise ValueError('Missing artifact: '+n)
    _write(out/'artifact_verification.json',dict(required_files=needed,all_present=True,remote_retention_verified=False))



def _archive(out,complete):
    items={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ('manifest.json','controller.lock') and p.suffix not in ('.tmp','.download') and p.name!='run.lock':items[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=_sha(p))
    _write(out/'manifest.json',dict(files=items,state='complete' if complete else ('training_complete_evaluation_pending' if (out/'training_only.json').exists() else 'partial'),remote_retention_verified=False,checkpoint_resume_requires='started arms latest.pt; queued unstarted arms initialize fresh; matching config/source/data'))
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
    print(('COMPLETE LOCAL OUTPUT: ' if complete else ('TRAINING OUTPUT, EVALUATION PENDING: ' if (out/'training_only.json').exists() else 'PARTIAL/CHECK OUTPUT: '))+str(zip_path),flush=True)
    print('This confirms local output only. Verify Saved Version Output and download its ZIP before ending storage.',flush=True)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(zip_path)));display(FileLink(str(zip_path.with_suffix('.zip.sha256'))))
    except ImportError:pass


def _run_wave(out,src,cp,cfg,wave,wave_index):
    handles=[];last={}
    print(json.dumps(dict(phase='wave_start',wave=wave_index+1,arms=list(wave),gpu_queue={str(i):[w[i] for w in WAVES[wave_index:]] for i in (0,1)},
          queue_eta='Unmeasured queued arm: total ETA pending; worker ETA covers current arm only'),ensure_ascii=False),flush=True)
    try:
        for arm in wave:
            if cfg['resume'] and cfg['mode']=='train':
                marker=out/'results'/arm/('complete.json' if cfg['evaluate_after_training'] else 'training_complete.json')
                if marker.exists():
                    print(arm+': already finished requested phase; skipping',flush=True);continue
            folder=out/'results'/arm;folder.mkdir(parents=True,exist_ok=True)
            removable=('failure.json','complete.json','status.json') if cfg['evaluate_after_training'] else ('failure.json','complete.json','training_complete.json','status.json')
            for name in removable:
                p=folder/name
                if p.exists():p.unlink()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['gpu_map'][arm]),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
            stream=(folder/'worker.log').open('a',encoding='utf8')
            try:proc=subprocess.Popen([sys.executable,'-u',str(src/'worker.py'),str(cp),arm],env=env,stdout=stream,stderr=subprocess.STDOUT)
            except BaseException:stream.close();raise
            handles.append((arm,proc,stream))
        checked=False
        while any(p.poll() is None for _,p,_ in handles):
            for arm,p,f in handles:
                path=out/'results'/arm/'status.json'
                if path.exists():
                    s=path.read_text()
                    if s!=last.get(arm):print(s,flush=True);last[arm]=s
            files=[out/'results'/a/'initialization.json' for a in wave]
            if not checked and all(p.exists() for p in files):
                checked=True
                if len({json.loads(p.read_text())['state_sha256'] for p in files})!=1:raise ValueError('Common initial state mismatch')
            if any(p.poll() is not None and p.returncode!=0 for _,p,_ in handles):raise RuntimeError('Worker failed; see captured worker log')
            time.sleep(5)
        for a,p,f in handles:
            if p.wait()!=0:raise RuntimeError(a+' worker failed')
            if cfg['mode']=='train':
                marker='complete.json' if cfg['evaluate_after_training'] else 'training_complete.json'
                if not(out/'results'/a/marker).exists():raise RuntimeError(a+' stopped before completion; resume required')
    except BaseException:
        for a,p,f in handles:
            if p.poll() is not None and p.returncode:
                print((out/'results'/a/'worker.log').read_text()[-10000:],flush=True)
        raise
    finally:
        for _,p,_ in handles:
            if p.poll() is None:p.terminate()
        deadline=time.monotonic()+45
        while any(p.poll() is None for _,p,_ in handles) and time.monotonic()<deadline:time.sleep(.2)
        for _,p,f in handles:
            if p.poll() is None:p.kill()
            p.wait();f.close()


def main():
    import importlib.util
    missing=[x for x in ('torch','h5py','numpy','scipy','skimage','PIL','matplotlib') if importlib.util.find_spec(x) is None]
    if missing:raise RuntimeError('Missing dependencies: '+str(missing))
    import torch
    if MODE not in ('train','check'):raise ValueError('MODE must be train/check')
    if not RUN_ID.replace('_','').replace('-','').isalnum() or '_s'+str(SEED)+'_' not in RUN_ID:raise ValueError('RUN_ID must be plain and include matching _s<SEED>_')
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):raise RuntimeError('Select GPU T4 x2; no CPU fallback')
    import fcntl
    out=Path('/kaggle/working')/RUN_ID
    lock=(out.parent/(RUN_ID+'.controller.lock')).open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resume=bool(RESUME_FROM)
    if resume:
        old=Path(RESUME_FROM)
        if MODE=='train' and (old/'config.json').exists() and json.loads((old/'config.json').read_text()).get('mode')=='check':
            raise ValueError('A check-only folder is not a training checkpoint; choose NEW RUN_ID and empty RESUME_FROM')
        if not(old/'config.json').is_file():raise FileNotFoundError('Exact previous run/config.json required')
        if (old/'manifest.json').is_file():
            for rel,info in json.loads((old/'manifest.json').read_text())['files'].items():
                p=old/rel
                if not p.is_file() or p.stat().st_size!=info['bytes'] or _sha(p)!=info['sha256']:raise ValueError('Previous manifest mismatch: '+rel)
        if old.resolve()!=out.resolve():
            if out.exists():raise FileExistsError('Resume destination exists')
            shutil.copytree(old,out)
    elif out.exists():raise FileExistsError('Existing run protected; set exact RESUME_FROM or choose NEW RUN_ID')
    out.mkdir(parents=True,exist_ok=True);src=out/'provenance/source'
    for name,content in _BUNDLE.items():
        p=src/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=content.encode():raise ValueError('Source changed; new run required')
        p.write_text(content,encoding='utf8')
    data={k:str(Path(DATA_ROOT)/v) for k,v in dict(train='QuickBird__Training_Dataset__train_qb.h5',val='QuickBird__Training_Dataset__valid_qb.h5',RR='QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5',FR='QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5').items()}
    manifests=_inspect(data)
    expected={'train':17139,'val':1905,'RR':20,'FR':20}
    import h5py
    for split,n in expected.items():
        with h5py.File(data[split]) as f:
            if len(f['pan'])!=n:raise ValueError('Unexpected sample count: '+split)
    executed,capture=_executed_source()
    cfg=dict(experiment=40,research_track='performance',experimenter='김경찬',out=str(out),data=data,seed=SEED,epochs=100,batch=64,micro=32,lr=1e-4,std_floor=.01,
        mode=MODE,resume=resume,evaluate_after_training=EVALUATE_AFTER_TRAINING,gpu_map={'A0':0,'A3':1,'A1':0,'A2':1},waves=WAVES,scenes=[1,8,11,13,19],
        controller_sha256=_controller_fingerprint(executed),source_hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in _BUNDLE.items()},
        source_parent='38 seed44 provenance/run_cell.py, PR99 head 56dcbc478d2ec7c47f30b49fb29bd13484de7602; common model/metrics/upstream byte-preserved',
        model='K4 original/LRPAN/BatchNorm maintained; A0 identity,A1 centering,A2 scaling,A3 centering+scaling',
        selection='minimum validation MSE including epoch0, earliest exact tie',precision='FP16 AMP train; FP32 transforms/loss/validation/test; TF32 off',
        normalization='DN/2047 no clipping; per-sample C,H,W MS population stats; GT excluded')
    stable={k:v for k,v in cfg.items() if k not in ('out','data','mode','resume','evaluate_after_training')}
    stable['data_hashes']={k:v['sha256'] for k,v in manifests.items()}
    cfg['fingerprint']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest();cp=out/'config.json'
    if resume and json.loads(cp.read_text())['fingerprint']!=cfg['fingerprint']:raise ValueError('Config/source/data mismatch: new run required')
    _write(cp,cfg);_write(out/'data_manifest.json',manifests)
    _write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpus=[torch.cuda.get_device_name(i) for i in (0,1)]))
    (out/'provenance/run_cell.py').write_text(executed,encoding='utf8')
    _write(out/'provenance/controller_capture.json',dict(method=capture,executed_source_sha256=hashlib.sha256(executed.encode()).hexdigest()))
    _write(out/'selection.json',dict(seed=SEED,indices=cfg['scenes'],index_base=0,unit='full RR scene',order=cfg['scenes'],rule='fixed before training/test',panels=['LR MS nearest','PAN','prediction','GT'],rgb=[2,1,0],contrast='shared per-scene GT 1/99 percentiles for MS panels',FR_has_GT=False))
    _write(out/'protocol.json',dict(experiment=40,status='plan, actual execution determined by markers',arms={'A0':'identity','A1':'x-mu / y+mu','A2':'x/s / y*s','A3':'(x-mu)/s / y*s+mu'},repeat_gate='A1 or A2 val MSE <=1.005*A3; decide before test',thresholds=dict(PSNR_min=-.03,SAM_max=.02,QNR_min=.003,Dlambda_max=0,Ds_max=0),test_tuning=False))
    if (out/'training_only.json').exists():(out/'training_only.json').unlink()
    if EVALUATE_AFTER_TRAINING:
        if not resume or not(out/'validation_decision.json').is_file():
            raise ValueError('Evaluation requires a completed validation-only run and exact RESUME_FROM; decide repeats BEFORE test')
        _verify_training(out,cfg)
    complete=False
    try:
        for wi,wave in enumerate(WAVES):_run_wave(out,src,cp,cfg,wave,wi)
        if MODE=='check':
            print('CHECK PASS: four GPU preflights; training NOT started. Use a NEW RUN_ID for train.',flush=True);return
        _verify_training(out,cfg);_training_plots(out)
        if not EVALUATE_AFTER_TRAINING:
            _write(out/'training_only.json',dict(state='training_complete_evaluation_pending',test_scored=False))
            print('TRAINING COMPLETE: inspect validation_decision.json BEFORE test. Resume same run with EVALUATE_AFTER_TRAINING=True to evaluate.',flush=True);return
        _plots(out);_verify_artifacts(out,cfg);complete=True
    finally:_archive(out,complete)

if __name__=='__main__':main()
