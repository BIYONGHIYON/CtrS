"""W2 feature high-frequency selective attenuation; common MSE training protocol."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, math, random, signal, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.nn import functional as F
from hf_gate_model import Model, load_common, common_state
from hf_gate_model import architecture_spec

STOP = False
KST = timezone(timedelta(hours=9))
KEYS = ('pan', 'lms', 'ms', 'gt')

def stop_handler(*_):
    global STOP
    STOP = True

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def atomic_save(path, value):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda:file.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def load_resident(config):
    """No hidden CPU fallback: train and validation stay on this worker's GPU."""
    result = {}
    required = sum(config['data'][s]['resident_fp32_bytes'] for s in ('train','val'))
    free, total = torch.cuda.mem_get_info()
    if free-required < config['reserve_gib']*(1<<30):
        raise RuntimeError(f'GPU resident data need {required/(1<<30):.2f} GiB; '
                           f'free={free/(1<<30):.2f}, reserve={config["reserve_gib"]} GiB. '
                           'No CPU fallback. Reduce BOTH model micro batches in a NEW run, '
                           'or use a documented new data-residency plan.')
    for split in ('train','val'):
        result[split] = {}
        with h5py.File(config['data'][split]['path'],'r') as h:
            n = config['data'][split]['used_samples']
            for key in KEYS:
                shape = (n,)+tuple(h[key].shape[1:])
                target = torch.empty(shape, device='cuda', dtype=torch.float32)
                # Chunked staging avoids holding the entire H5 twice in CPU RAM.
                for start in range(0,n,64):
                    array = np.asarray(h[key][start:min(n,start+64)],dtype=np.float32)
                    if not np.isfinite(array).all():
                        raise ValueError(f'{split}/{key}: nonfinite data')
                    target[start:start+len(array)].copy_(torch.from_numpy(array / 2047.))
                result[split][key] = target
    torch.cuda.synchronize()
    return result

def validate(model,data,batch):
    """Unmodified inputs; full archived33 MSE/PSNR/SAM, fixed first32 auxiliary metrics."""
    from common.metrics import rr_metrics
    model.eval();s=torch.zeros(4,device='cuda',dtype=torch.float64);inner_elements=0
    pixels=torch.zeros((),device='cuda',dtype=torch.int64);n=0;elements=0;rows=[]
    with torch.inference_mode():
        for start in range(0,len(data['pan']),batch):
            b={k:v[start:start+batch] for k,v in data.items()};gt=b['gt']
            p=model(b['pan'],b['lms'],b['ms']).float()
            if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite validation')
            error=(p-gt).square();s[0]+=error.double().sum();elements+=gt.numel()
            inner=error[...,4:-4,4:-4];s[3]+=inner.double().sum();inner_elements+=inner.numel()
            s[1]+=(-10*torch.log10(error.mean((2,3)).clamp_min(1e-15))).mean(1).double().sum();n+=len(gt)
            norm=p.norm(dim=1)*gt.norm(dim=1);mask=norm>1e-12
            cos=(p*gt).sum(1)/norm.clamp_min(1e-12)
            s[2]+=torch.acos(cos.clamp(-1,1))[mask].double().sum();pixels+=mask.sum()
            if start<32:
                a=p[:32-start].cpu().numpy().astype(np.float64)*2047.;g=gt[:32-start].cpu().numpy().astype(np.float64)*2047.
                rows.extend(rr_metrics(x,y) for x,y in zip(g,a))
    a=s.cpu().tolist();model.train()
    return {'mse':a[0]/elements,'psnr':a[1]/n,'sam':math.degrees(a[2]/max(int(pixels),1)),
            'mse_inner4':a[3]/inner_elements,'mse_border4':(a[0]-a[3])/(elements-inner_elements),
            **{k.lower():float(np.mean([r[k] for r in rows])) for k in ('ERGAS','SCC','Q2n')},
            'evaluation_policy':{'applied':False,'scope':'unmodified validation inputs; no training or test-time augmentation'}}










def panels(arrays, pred, path, scene, protocol='RR full scene'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb_bands = [2,1,0]
    ref = arrays['gt'][rgb_bands].transpose(1,2,0)
    low, high = np.percentile(ref,(1,99),axis=(0,1))
    def rgb(a):
        return np.clip((a[rgb_bands].transpose(1,2,0)-low)/np.maximum(high-low,1e-8),0,1)
    fig,axes = plt.subplots(1,4,figsize=(16,4))
    images = [rgb(arrays['ms']),arrays['pan'][0],rgb(pred),rgb(arrays['gt'])]
    for ax,title,image in zip(axes,('LR MS (nearest)','PAN input','Prediction','GT'),images):
        ax.imshow(image, interpolation='nearest', cmap='gray' if image.ndim==2 else None)
        ax.set_title(title); ax.axis('off')
    fig.suptitle(f'{protocol} {scene} | same GT-derived MS contrast')
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)

def evaluate(config, folder, model, report):
    import sys
    sys.path.insert(0,str(Path(config['code_root'])/'SSA-MRN/src'))
    from common.metrics import rr_metrics, fr_metrics
    eval_name=config.get('eval_checkpoint','best')
    ck = torch.load(folder/(eval_name+'.pt'),map_location='cpu',weights_only=False)
    if ck['fingerprint'] != config['fingerprint']:
        raise ValueError('Best fingerprint mismatch')
    model.load_state_dict(ck['model'],strict=True); model.eval()
    checkpoint_folder=folder
    folder=folder/'evaluations'/eval_name; folder.mkdir(parents=True,exist_ok=True)
    results = {'best_epoch':ck['epoch'], 'best_val_mse':ck.get('mse',ck.get('best_mse')), 'full_scene':True,
               'test_precision':'FP32', 'checkpoint_kind':eval_name, 'limitations':'FR D_s/QNR MATLAB parity is unverified; RR Q2n/SCC parity is unverified.',
               'splits':{}}
    from common.evaluation_protocol import WaldDegradation,observation_rmse
    from common.observation import SensorObservation
    smoke=bool(config.get('smoke_evaluation'))
    degrade=WaldDegradation('QB','cuda') if not smoke else None
    observation=SensorObservation('QB','decimate',(2,2)).cuda().double() if not smoke else None
    inference_times=[];warmed_forward={};gate_records={}
    for split in (('RR','FR') if smoke else ('RR','RR2','FR_WALD','FR')):
        rows=[]; tick=time.monotonic()
        # Smoke checks use validation inputs only; held-out RR/FR are never scored here.
        entry=config['data']['val'] if config.get('smoke_evaluation') else config['data']['RR' if split=='RR2' else ('FR' if split=='FR_WALD' else split)]
        with h5py.File(entry['path'],'r') as h:
            indices=range(min(2,len(h['pan']))) if config.get('smoke_evaluation') else range(len(h['pan']))
            for index in indices:
                report(stage='evaluate_'+split,scene=index+1,scenes=len(h['pan']),
                       train_eta_kst=None,total_eta_kst=None,eta_scope='test scoring/I/O unmeasured')
                keys = KEYS if split=='RR' else KEYS[:3]
                arrays={k:np.asarray(h[k][index],dtype=np.float64) for k in keys}
                if split in ('RR2','FR_WALD'):
                    gt=arrays['ms'].copy()
                    pp,ll,mm=degrade(arrays['pan'],arrays['ms'])
                    arrays={'pan':pp,'lms':ll,'ms':mm,'gt':gt}
                inputs=[torch.from_numpy((arrays[k]/2047.).astype(np.float32)).unsqueeze(0).cuda() for k in KEYS[:3]]
                if index==0:
                    with torch.inference_mode():
                        for _ in range(3):model(*inputs)
                        probe=[]
                        for _ in range(5):
                            torch.cuda.synchronize();t=time.monotonic();model(*inputs);torch.cuda.synchronize();probe.append(time.monotonic()-t)
                    warmed_forward[split]={'median_ms':1000*float(np.median(probe)),'warmup':3,'repetitions':5,'scope':'first scene only; transfer/metrics excluded'}
                torch.cuda.synchronize(); infer_tick=time.monotonic()
                with torch.inference_mode():
                    gpu_pred=model(*inputs)[0].float()
                torch.cuda.synchronize(); infer_seconds=time.monotonic()-infer_tick
                inference_times.append({'split':split,'scene_index':index,'seconds':infer_seconds,
                                        'scope':'FP32 forward only; first-scene probe performed; transfer/metrics excluded'})
                pred=gpu_pred.cpu().numpy().astype(np.float64)*2047.
                if not np.isfinite(pred).all():
                    raise RuntimeError('Nonfinite test prediction')
                metrics=rr_metrics(arrays['gt'],pred) if split!='FR' else fr_metrics(pred,arrays['lms'],arrays['ms'],arrays['pan'])
                if split=='FR' and not smoke:metrics['OBS_RMSE_DN']=observation_rmse(pred,arrays['ms'],observation)
                gate_records.setdefault(split,[]).append({'scene_index':index,**model.gate_snapshot()})
                if split=='RR':
                    err=((pred.astype(np.float64)-arrays['gt'])/2047.)**2
                    metrics['MSE_peak1']=float(err.mean())
                    inner=err[...,4:-4,4:-4]
                    metrics['MSE_inner4_peak1']=float(inner.mean())
                    metrics['MSE_border4_peak1']=float((err.sum()-inner.sum())/(err.size-inner.size))
                    if index in config['selection']['indices']:
                        panels(arrays,pred,folder/f'RR_scene_{index:03}.png',index)
                if not all(math.isfinite(float(v)) for v in metrics.values()):
                    raise RuntimeError('Nonfinite test metric')
                rows.append({'scene_index':index, **metrics})
                atomic_json(folder/f'{split}_per_scene.json',rows)
                del inputs
        results['splits'][split]={'count':len(rows),'seconds':time.monotonic()-tick,
                                 'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='scene_index'}}
    results['smoke_validation_inputs_only']=bool(config.get('smoke_evaluation'))
    results['gate_records']=gate_records
    results['criteria_version']='v2.2'
    results['inference_times']=inference_times
    results['warmed_forward']=warmed_forward
    results['test_input_policy']='unmodified native inputs; RR2/FR_WALD use pinned evaluation protocol; only W2 frequency-selective gate differs'
    results['peak_gpu_gib']=torch.cuda.max_memory_allocated()/(1<<30)
    results['checkpoints']={k:{'sha256':file_hash(checkpoint_folder/f'{k}.pt')} for k in ('best','latest')}
    results['evaluation_protocol_sha256']=file_hash(Path(config['code_root'])/'common/evaluation_protocol.py')
    atomic_json(folder/'results.json',results)
    atomic_json(folder/'evaluation_complete.json',{'RR':results['splits']['RR']['count'],'FR':results['splits']['FR']['count'],
                'fingerprint':config['fingerprint'],'best_sha256':file_hash(checkpoint_folder/'best.pt'),
                'latest_sha256':file_hash(checkpoint_folder/'latest.pt'),'results_sha256':file_hash(folder/'results.json')})

def train(config, label, physical_gpu):
    import sys
    sys.path.insert(0,str(Path(config['code_root'])/'SSA-MRN/src'))
    from hf_gate_model import Model
    folder=Path(config['output'])/label; folder.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGINT,stop_handler)
    assert torch.cuda.device_count()==1, 'Each process must see exactly one GPU'
    global STOP
    STOP = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=not config.get('smoke_deterministic',False))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    def report(**kw):
        value={'label':label,'physical_gpu':physical_gpu,'k':4,'total_epochs':config['epochs'],
               'updated_kst':datetime.now(KST).isoformat(),
               'allocated_gib':torch.cuda.memory_allocated()/(1<<30),
               'reserved_gib':torch.cuda.memory_reserved()/(1<<30),
               'peak_gib':torch.cuda.max_memory_allocated()/(1<<30),**kw}
        atomic_json(folder/'status.json',value)
        with open(folder/'progress.jsonl','a',encoding='utf-8') as file:
            file.write(json.dumps(value,allow_nan=False)+'\n')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    report(stage='load_gpu_data',train_eta_kst=None,total_eta_kst=None,eta_scope='measurement pending')
    data=load_resident(config) if config.get('job_type')!='test' else None
    model=Model(config['arm'],seed=config['seed']).cuda()
    initial=torch.load(Path(config['output'])/'initial.pt',map_location='cpu',weights_only=True)
    load_common(model,initial)
    initial_hash=state_hash(model.state_dict())
    atomic_json(folder/'initialization.json',{'state_sha256':initial_hash,
        'common_state_sha256':state_hash(common_state(model)), 'model':'43_C2_W2_with_same_inactive_or_active_gate',
        'parameters':sum(p.numel() for p in model.parameters()), 'initial_outputs_expected_equal':True,'spatial_operator':architecture_spec(config['arm']), 'k':4})
    opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
    stepped=[False]
    opt.register_step_post_hook(lambda *_:stepped.__setitem__(0,True))
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'],init_scale=1024.)
    generator=torch.Generator(device='cuda').manual_seed(config['seed']+100000) # archived33 stream; separate model/sampler RNG
    n=len(data['train']['gt']) if data is not None else 0; batch=config['batch']; micro=config['micro']
    groups=math.ceil(n/batch)
    epoch=0; position=0; order=None; order_hash=None; best=None; best_payload=None
    history=[]; orders=[]; timings=[]
    total=torch.zeros((),device='cuda',dtype=torch.float64); count=0; skipped=0
    # Save only at completed optimizer-step boundaries; no partial gradients.
    active_train_seconds=0.; training_timer=None
    def checkpoint():
        nonlocal last_save, active_train_seconds, training_timer
        if training_timer is not None:
            torch.cuda.synchronize()
            active_train_seconds+=time.monotonic()-training_timer
            training_timer=time.monotonic()
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state(),'generator':generator.get_state()}
        state={'fingerprint':config['fingerprint'],'config':config['conditions'],
               'model':cpu_state(model),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
               'lr_schedule':{'type':'constant','lr':config['lr']},'rng':rng,
               'epoch':epoch,'next_sample_position':position,'order':None if order is None else order.cpu(),
               'order_hash':order_hash,'order_history':orders,'loss_sse':total.cpu(),'loss_elements':count,
               'history':history,'timings':timings,'best_mse':best,'best_payload':best_payload,
               'initial_state_sha256':initial_hash,'skipped_amp_steps':skipped,
               'active_train_seconds':active_train_seconds}
        atomic_save(folder/'latest.pt',state)
        # Preserve the actual historical best, even at non-improving epochs.
        if best_payload is not None:
            atomic_save(folder/'best.pt',best_payload)
        atomic_json(folder/'history.json',history)
        atomic_json(folder/'orders.json',orders)
        last_save=time.monotonic()
    if config.get('job_type')=='test':
        for kind in ('best','latest'):
            evaluate(dict(config,eval_checkpoint=kind),folder,model,report)
        atomic_json(folder/'evaluation_complete.json',{'fingerprint':config['fingerprint'],
            'best_sha256':file_hash(folder/'best.pt'),'latest_sha256':file_hash(folder/'latest.pt'),
            'results_sha256':{k:file_hash(folder/'evaluations'/k/'results.json') for k in ('best','latest')}})
        report(stage='evaluation_complete')
        return
    if (folder/'latest.pt').exists():
        ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if ck['fingerprint']!=config['fingerprint'] or ck['initial_state_sha256']!=initial_hash:
            raise ValueError('Resume conditions/initial state mismatch; use a new run')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        epoch=ck['epoch']; position=ck['next_sample_position']; order=ck['order']
        if order is not None: order=order.cuda()
        order_hash=ck['order_hash']; orders=ck['order_history']; history=ck['history']; timings=ck['timings']
        total=ck['loss_sse'].cuda(); count=ck['loss_elements']; best=ck['best_mse']; best_payload=ck['best_payload']
        skipped=ck['skipped_amp_steps']
        active_train_seconds=ck['active_train_seconds']
        r=ck['rng']; random.setstate(r['python']); np.random.set_state(r['numpy'])
        torch.set_rng_state(r['cpu']); torch.cuda.set_rng_state(r['cuda']); generator.set_state(r['generator'])
    # Include epoch0 in best selection exactly as archived33. Never choose using test.
    if best is None:
        initial_val=validate(model,data['val'],micro)
        best=initial_val['mse']
        best_payload={'fingerprint':config['fingerprint'],'model':cpu_state(model),'epoch':0,
                      'mse':best,'validation':initial_val,'config':config['conditions']}
        history=[{'epoch':0,'lr':config['lr'],'train_mse':None,'train_total':None,
                  **{'val_'+k:v for k,v in initial_val.items() if k!='evaluation_policy'},
                  'evaluation_policy':initial_val['evaluation_policy'],'validation_diagnostic_scope':'ERGAS/SCC/Q2n: fixed first32 val; MSE/PSNR/SAM: full used val'}]
    last_save=time.monotonic(); checkpoint()
    last_log=0.; model.train()
    while epoch<config['epochs']:
        if STOP or time.time()>=config['deadline']:
            checkpoint(); report(stage='paused',epoch=epoch,next_sample_position=position); return
        tick=time.monotonic(); training_timer=tick; segment_start=position; steps_done=0
        if order is None:
            order=torch.randperm(n,device='cuda',generator=generator)
            order_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            orders.append({'epoch':epoch+1,'order_sha256':order_hash})
            atomic_json(folder/'orders.json',orders)
            checkpoint()
        for start in range(position,n,batch):
            end=min(n,start+batch); opt.zero_grad(set_to_none=True)
            group_ids=order[start:end]
            for offset in range(start,end,micro):
                ids=order[offset:min(end,offset+micro)]
                b={k:v.index_select(0,ids) for k,v in data['train'].items()}
                with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                    pred=model(b['pan'],b['lms'],b['ms'])
                squared=(pred.float()-b['gt']).square().sum() # FP32 outside autocast, archived33 operation
                # Deliberately no per-batch .item()/CPU transfers.
                total+=squared.detach().double()
                count+=b['gt'].numel()
                group_loss=squared/((end-start)*data['train']['gt'][0].numel())
                scaler.scale(group_loss).backward()
            stepped[0]=False
            scaler.step(opt); scaler.update()
            skipped+=int(not stepped[0])
            position=end; steps_done+=1
            if time.monotonic()-last_save>=60:
                checkpoint()
            if STOP or time.time()>=config['deadline'] or (config.get('stop_after_group') and math.ceil(position/batch)>=config['stop_after_group']):
                checkpoint(); report(stage='paused',epoch=epoch,next_sample_position=position); return
            if time.monotonic()-last_log>=config['log_seconds']:
                torch.cuda.synchronize(); elapsed=time.monotonic()-tick
                remaining_groups=math.ceil((n-position)/batch)
                remainder=remaining_groups*elapsed/max(steps_done,1)
                eta=None
                if timings:
                    remainder+=float(np.mean([t['validation_save_seconds'] for t in timings]))
                    remainder+=(config['epochs']-epoch-1)*float(np.mean([t['epoch_wall_seconds'] for t in timings]))
                    eta=(datetime.now(KST)+timedelta(seconds=remainder)).isoformat()
                mse=float(total.cpu())/max(count,1)
                if not math.isfinite(mse): raise RuntimeError('Nonfinite train MSE')
                report(stage='train',epoch=epoch+1,batch=math.ceil(position/batch),batches=groups,
                       lr=opt.param_groups[0]['lr'],train_mse=mse,train_total=mse,elapsed_seconds=elapsed,
                       train_eta_kst=eta,total_eta_kst=None,
                       eta_scope='measured train+val+save; test scoring, figures and ZIP excluded' if eta else 'first epoch measurement pending')
                last_log=time.monotonic()
        torch.cuda.synchronize(); train_time=time.monotonic()-tick
        active_train_seconds+=time.monotonic()-training_timer; training_timer=None
        report(stage='validation',epoch=epoch+1,train_eta_kst=None,total_eta_kst=None)
        val_tick=time.monotonic(); val=validate(model,data['val'],micro)
        torch.cuda.synchronize(); val_seconds=time.monotonic()-val_tick
        mse=float(total.cpu())/count
        if not math.isfinite(mse) or not all(math.isfinite(v) for k,v in val.items() if k!='evaluation_policy'):
            raise RuntimeError('Nonfinite epoch statistics')
        row={'lr':opt.param_groups[0]['lr'],'epoch':epoch+1,'train_mse':mse,'val_mse':val['mse'],'val_psnr':val['psnr'],'val_sam':val['sam'],
             'skipped_amp_steps':skipped,'order_sha256':order_hash,'train_segment_seconds':train_time,
             'validation_seconds':val_seconds,'resumed_mid_epoch':segment_start>0,
             'train_total':mse,
             'val_ergas':val['ergas'],'val_scc':val['scc'],'val_q2n':val['q2n']}
        row['train_total']=mse
        row['validation_diagnostic_scope']='ERGAS/SCC/Q2n: first 32 validation patches; MSE/PSNR/SAM: all used validation patches'
        row['evaluation_policy']=val['evaluation_policy']
        row['val_mse_inner4']=val['mse_inner4'];row['val_mse_border4']=val['mse_border4']
        history.append(row)
        if best is None or val['mse']<best:
            best=val['mse']; best_payload={'fingerprint':config['fingerprint'],'model':cpu_state(model),
                                         'epoch':epoch+1,'mse':best,'validation':val,'config':config['conditions']}
        epoch+=1; position=0; order=None; order_hash=None; total.zero_();count=0
        save_tick=time.monotonic(); checkpoint(); save_seconds=time.monotonic()-save_tick
        # Mid-epoch resumed segments are excluded from epoch-based ETA calibration.
        if segment_start==0:
            timings.append({'epoch_wall_seconds':time.monotonic()-tick,
                            'validation_save_seconds':val_seconds+save_seconds})
        atomic_json(folder/'history.json',history)
        remaining=(config['epochs']-epoch)*float(np.mean([t['epoch_wall_seconds'] for t in timings])) if timings else None
        report(stage='epoch_complete',**row,train_eta_kst=(datetime.now(KST)+timedelta(seconds=remaining)).isoformat() if remaining is not None else None,
               total_eta_kst=None,eta_scope='test scoring, figures and ZIP excluded')
    checkpoint()
    completion_proof={'epochs':epoch,'best_epoch':best_payload['epoch'],'best_mse':best,
        'fingerprint':config['fingerprint'],'best_sha256':file_hash(folder/'best.pt'),'latest_sha256':file_hash(folder/'latest.pt')}
    # Free resident training arrays before full-scene FP32 evaluation.
    del data, opt, scaler; torch.cuda.empty_cache()
    atomic_json(folder/'validation_result.json',{'best_epoch':best_payload['epoch'],
                'best_validation':best_payload['validation'],'last_validation':history[-1],
                'last_epoch':epoch,'peak_gpu_gib':torch.cuda.max_memory_allocated()/(1<<30),
                'train_seconds_measured_segments':sum(h.get('train_segment_seconds',0.) for h in history),
                'train_active_seconds':active_train_seconds,
                'timing_limit':'active train timer includes within-epoch checkpoint overhead; excludes data loading/validation; preserved across resume'})
    atomic_json(folder/'training_complete.json',completion_proof)
    report(stage='training_complete_no_test',epoch=epoch,train_eta_kst=datetime.now(KST).isoformat(),total_eta_kst=None)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config'); parser.add_argument('--label'); parser.add_argument('--gpu',type=int)
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text())
    try:
        train(config,args.label,args.gpu)
    except Exception as exc:
        folder=Path(config['output'])/args.label; folder.mkdir(exist_ok=True)
        atomic_json(folder/'failure.json',{'error':str(exc),'type':type(exc).__name__})
        raise

if __name__=='__main__':
    main()
