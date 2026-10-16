"""SpectralLoss_1st worker derived from the validated A/A harness; FP32 fixed baseline."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, math, random, signal, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.nn import functional as F
from gate29_model import W2Net, load_common, common_state

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

def validate(model, data, batch):
    """Match 29 primary full-validation statistics; extra quality diagnostics on a fixed subset."""
    from common.metrics import rr_metrics
    model.eval()
    sums=torch.zeros(7,device='cuda',dtype=torch.float64)
    gate_sum=torch.zeros(2,device='cuda',dtype=torch.float64);gate_n=0
    gate_min=torch.ones((),device='cuda');gate_max=torch.zeros((),device='cuda')
    rows=[]; n=len(data['gt'])
    with torch.inference_mode():
        for start in range(0,n,batch):
            b={k:v[start:start+batch] for k,v in data.items()}
            pred=model(b['pan'],b['lms'],b['ms']).float()
            if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite validation')
            diff=(pred-b['gt']).square()
            # Sum in FP32 before double accumulation, as in archived 29.
            sums[0]+=diff.sum().double();sums[1]+=diff.numel()
            bmse=diff.mean((2,3));sums[2]+=(-10*torch.log10(bmse.clamp_min(1e-15))).mean(1).sum().double()
            norms=(pred.norm(dim=1)*b['gt'].norm(dim=1)).clamp_min(1e-12)
            valid=(b['gt'].norm(dim=1)>1e-8)&(pred.norm(dim=1)>1e-8)
            sums[3]+=torch.acos(((pred*b['gt']).sum(1)/norms).clamp(-1,1))[valid].sum().double()
            sums[4]+=valid.sum()
            g=getattr(model.net.backbone,'last_gate',None)
            if g is not None:
                gate_sum[0]+=g.sum().double();gate_sum[1]+=g.square().sum().double();gate_n+=g.numel()
                gate_min=torch.minimum(gate_min,g.min());gate_max=torch.maximum(gate_max,g.max())
            if start<32:
                p=pred[:32-start].cpu().numpy().astype(np.float64)*2047.
                gt=b['gt'][:32-start].cpu().numpy().astype(np.float64)*2047.
                rows.extend(rr_metrics(a,c) for a,c in zip(gt,p))
    v=sums.cpu().tolist();gate={}
    if gate_n:
        gs=gate_sum.cpu().tolist();mean=gs[0]/gate_n
        gate={'mean':mean,'std':math.sqrt(max(0,gs[1]/gate_n-mean*mean)),
              'min':float(gate_min.cpu()),'max':float(gate_max.cpu())}
    model.train()
    return {'mse':v[0]/v[1], 'psnr':v[2]/n, 'sam':math.degrees(v[3]/max(v[4],1)),
            **{k.lower():float(np.mean([r[k] for r in rows])) for k in ('ERGAS','SCC','Q2n')},'gate':gate}


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
    inference_times=[]
    for split in ('RR','FR'):
        rows=[]; tick=time.monotonic()
        # Smoke checks use validation inputs only; held-out RR/FR are never scored here.
        entry=config['data']['val'] if config.get('smoke_evaluation') else config['data'][split]
        with h5py.File(entry['path'],'r') as h:
            indices=range(min(2,len(h['pan']))) if config.get('smoke_evaluation') else range(len(h['pan']))
            for index in indices:
                report(stage='evaluate_'+split,scene=index+1,scenes=len(h['pan']),
                       train_eta_kst=None,total_eta_kst=None,eta_scope='test scoring/I/O unmeasured')
                keys = KEYS if split=='RR' else KEYS[:3]
                arrays={k:np.asarray(h[k][index],dtype=np.float32) for k in keys}
                inputs=[torch.from_numpy(arrays[k]/2047.).unsqueeze(0).cuda() for k in KEYS[:3]]
                torch.cuda.synchronize(); infer_tick=time.monotonic()
                with torch.inference_mode():
                    gpu_pred=model(*inputs)[0].float()
                torch.cuda.synchronize(); infer_seconds=time.monotonic()-infer_tick
                inference_times.append({'split':split,'scene_index':index,'seconds':infer_seconds,
                                        'scope':'FP32 forward only; transfer/metrics excluded; no warmup exclusion'})
                pred=gpu_pred.cpu().numpy().astype(np.float64)*2047.
                if not np.isfinite(pred).all():
                    raise RuntimeError('Nonfinite test prediction')
                metrics=rr_metrics(arrays['gt'],pred) if split=='RR' else fr_metrics(pred,arrays['lms'],arrays['ms'],arrays['pan'])
                if split=='RR':
                    metrics['MSE_peak1']=float(np.mean(((pred.astype(np.float64)-arrays['gt'])/2047.)**2))
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
    results['inference_times']=inference_times
    results['peak_gpu_gib']=torch.cuda.max_memory_allocated()/(1<<30)
    results['checkpoints']={k:{'sha256':file_hash(checkpoint_folder/f'{k}.pt')} for k in ('best','latest')}
    atomic_json(folder/'results.json',results)
    atomic_json(folder/'evaluation_complete.json',{'RR':results['splits']['RR']['count'],'FR':results['splits']['FR']['count'],
                'fingerprint':config['fingerprint'],'best_sha256':file_hash(checkpoint_folder/'best.pt'),
                'latest_sha256':file_hash(checkpoint_folder/'latest.pt'),'results_sha256':file_hash(folder/'results.json')})

def train(config, label, physical_gpu):
    import sys
    sys.path.insert(0,str(Path(config['code_root'])/'SSA-MRN/src'))
    from gate29_model import W2Net
    folder=Path(config['output'])/label; folder.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGINT,stop_handler)
    assert torch.cuda.device_count()==1, 'Each process must see exactly one GPU'
    global STOP
    STOP = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(config.get('smoke_deterministic', False))
    torch.backends.cudnn.benchmark=not config.get('smoke_deterministic', False)
    torch.backends.cudnn.deterministic=config.get('smoke_deterministic', False)
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
    model=W2Net(gated=config['gated']).cuda()
    initial=torch.load(Path(config['output'])/'initial.pt',map_location='cpu',weights_only=True)
    load_common(model,initial)
    initial_hash=state_hash(model.state_dict())
    atomic_json(folder/'initialization.json',{'state_sha256':initial_hash,
        'common_state_sha256':state_hash(common_state(model)), 'model':'W2_gated' if config['gated'] else 'W2',
        'parameters':sum(p.numel() for p in model.parameters()), 'initial_gate':0.5 if config['gated'] else 1.0, 'k':4})
    opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
    stepped=[False]
    opt.register_step_post_hook(lambda *_:stepped.__setitem__(0,True))
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'])
    generator=torch.Generator().manual_seed(config['seed']) # CPU permutation, separate from model RNG
    n=len(data['train']['gt']) if data is not None else 0; batch=config['batch']; micro=config['micro']
    groups=math.ceil(n/batch)
    epoch=0; position=0; order=None; order_hash=None; best=None; best_payload=None
    history=[]; orders=[]; timings=[]
    total=torch.zeros((),device='cuda',dtype=torch.float64); count=0; skipped=0
    spec_total=torch.zeros((),device='cuda',dtype=torch.float64)
    spec_count=torch.zeros((),device='cuda',dtype=torch.float64)
    objective_total=torch.zeros((),device='cuda',dtype=torch.float64)
    objective_samples=0
    objective_weight=0.0 # MSE only; no auxiliary objective
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
               'lr_schedule':{'type':'cosine','start':1e-4,'end':1e-6,'denominator':149},'rng':rng,
               'epoch':epoch,'next_sample_position':position,'order':None if order is None else order.cpu(),
               'order_hash':order_hash,'order_history':orders,'loss_sse':total.cpu(),'loss_elements':count,
               'history':history,'timings':timings,'best_mse':best,'best_payload':best_payload,
               'initial_state_sha256':initial_hash,'skipped_amp_steps':skipped,
               'spec_total':spec_total.cpu(),'spec_count':spec_count.cpu(),
               'objective_total':objective_total.cpu(),'objective_samples':objective_samples,'active_train_seconds':active_train_seconds}
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
        spec_total=ck['spec_total'].cuda(); spec_count=ck['spec_count'].cuda()
        objective_total=ck['objective_total'].cuda(); objective_samples=ck['objective_samples']; active_train_seconds=ck['active_train_seconds']
        r=ck['rng']; random.setstate(r['python']); np.random.set_state(r['numpy'])
        torch.set_rng_state(r['cpu']); torch.cuda.set_rng_state(r['cuda']); generator.set_state(r['generator'])
    last_save=time.monotonic(); checkpoint()
    last_log=0.; model.train()
    while epoch<config['epochs']:
        if STOP or time.time()>=config['deadline']:
            checkpoint(); report(stage='paused',epoch=epoch,next_sample_position=position); return
        for group in opt.param_groups:
            group['lr']=1e-6+(1e-4-1e-6)*0.5*(1+math.cos(math.pi*epoch/149))
        tick=time.monotonic(); training_timer=tick; segment_start=position; steps_done=0
        if order is None:
            order=torch.randperm(n,generator=torch.Generator().manual_seed(config['seed']+epoch)).cuda()
            order_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            orders.append({'epoch':epoch+1,'order_sha256':order_hash})
            atomic_json(folder/'orders.json',orders)
            checkpoint()
        for start in range(position,n,batch):
            end=min(n,start+batch); opt.zero_grad(set_to_none=True)
            group_ids=order[start:end]
            group_valid=1
            group_obj=torch.zeros((),device='cuda',dtype=torch.float64)
            for offset in range(start,end,micro):
                ids=order[offset:min(end,offset+micro)]
                b={k:v.index_select(0,ids) for k,v in data['train'].items()}
                with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                    pred=model(b['pan'],b['lms'],b['ms'])
                    loss=(pred.float()-b['gt']).square().sum()/b['gt'].numel()
                # Deliberately no per-batch .item()/CPU transfers.
                total+=(pred.detach().float()-b['gt']).square().sum(dtype=torch.float64)
                count+=b['gt'].numel()
                spec_sum=torch.zeros((),device='cuda'); valid_count=torch.ones((),device='cuda')
                spec_total+=spec_sum.detach().double(); spec_count+=valid_count.double()
                group_loss=loss*len(ids)/(end-start)
                if objective_weight:
                    group_loss=group_loss+objective_weight*spec_sum/group_valid
                group_obj+=group_loss.detach().double()
                scaler.scale(group_loss).backward()
            objective_total+=group_obj*(end-start); objective_samples+=end-start
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
                       lr=opt.param_groups[0]['lr'],train_mse=mse,train_spectral=float(spec_total.cpu()/spec_count.clamp_min(1)),
                       train_total=float(objective_total.cpu())/max(objective_samples,1),weight=objective_weight,elapsed_seconds=elapsed,
                       train_eta_kst=eta,total_eta_kst=None,
                       eta_scope='measured train+val+save; test scoring, figures and ZIP excluded' if eta else 'first epoch measurement pending')
                last_log=time.monotonic()
        torch.cuda.synchronize(); train_time=time.monotonic()-tick
        active_train_seconds+=time.monotonic()-training_timer; training_timer=None
        report(stage='validation',epoch=epoch+1,train_eta_kst=None,total_eta_kst=None)
        val_tick=time.monotonic(); val=validate(model,data['val'],micro)
        torch.cuda.synchronize(); val_seconds=time.monotonic()-val_tick
        mse=float(total.cpu())/count
        if not math.isfinite(mse) or not all(math.isfinite(v) for k,v in val.items() if k!='gate'):
            raise RuntimeError('Nonfinite epoch statistics')
        row={'lr':opt.param_groups[0]['lr'],'epoch':epoch+1,'train_mse':mse,'val_mse':val['mse'],'val_psnr':val['psnr'],'val_sam':val['sam'],
             'skipped_amp_steps':skipped,'order_sha256':order_hash,'train_segment_seconds':train_time,
             'validation_seconds':val_seconds,'resumed_mid_epoch':segment_start>0,
             'train_spectral':float(spec_total.cpu()/spec_count.clamp_min(1)),
             'train_total':float(objective_total.cpu())/objective_samples,'lambda':objective_weight,
             'val_ergas':val['ergas'],'val_scc':val['scc'],'val_q2n':val['q2n']}
        for unused in ('train_spectral','lambda'):
            row.pop(unused,None)
        row['train_total']=mse
        row['validation_diagnostic_scope']='ERGAS/SCC/Q2n: first 32 validation patches; MSE/PSNR/SAM: all used validation patches'
        row['gate']=val.get('gate',{})
        history.append(row)
        if best is None or val['mse']<best:
            best=val['mse']; best_payload={'fingerprint':config['fingerprint'],'model':cpu_state(model),
                                         'epoch':epoch+1,'mse':best,'validation':val,'config':config['conditions']}
        epoch+=1; position=0; order=None; order_hash=None; total.zero_(); count=0
        spec_total.zero_(); spec_count.zero_(); objective_total.zero_(); objective_samples=0
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
                'train_seconds_measured_segments':sum(h['train_segment_seconds'] for h in history),
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
