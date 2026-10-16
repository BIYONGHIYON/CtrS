import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,math,hashlib,random,copy,signal,traceback,platform
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h5py
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from normalization import Model
from common.metrics import rr_metrics,fr_metrics
KEYS=('pan','lms','ms','gt')
KST=timezone(timedelta(hours=9))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def digest_state(sd):
    h=hashlib.sha256()
    for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)

def cpu(obj):
    if torch.is_tensor(obj):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:cpu(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu(v) for v in obj)
    return obj

def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    torch.save(cpu(obj),t)
    with t.open('r+b') as f:f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def read_arrays(path,gt=True):
    with h5py.File(path,'r') as f:return {k:np.asarray(f[k],dtype=np.float32)/2047. for k in (KEYS if gt else KEYS[:3])}

def gpu_cache(path):
    arrays=read_arrays(path)
    required=sum(v.nbytes for v in arrays.values())
    free,total=torch.cuda.mem_get_info()
    if required>free*0.8:raise MemoryError(f'GPU resident cache requires {required/2**30:.2f} GiB; free {free/2**30:.2f} GiB. No silent CPU fallback.')
    return {k:torch.from_numpy(a).cuda() for k,a in arrays.items()}

@torch.inference_mode()
def validation(m,cache,micro):
    m.eval();s=torch.zeros(3,device='cuda',dtype=torch.float64)
    pixels=torch.zeros((),device='cuda',dtype=torch.int64);n=0;elements=0
    for i in range(0,len(cache['pan']),micro):
        b=[cache[k][i:i+micro] for k in KEYS];gt=b[3]
        p=m(*b[:3]).float()
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite validation prediction')
        error=(p-gt).square()
        s[0]+=error.double().sum();elements+=gt.numel()
        s[1]+=(-10*torch.log10(error.mean((2,3)).clamp_min(1e-15))).mean(1).double().sum();n+=len(gt)
        norm=p.norm(dim=1)*gt.norm(dim=1);mask=norm>1e-12
        cos=(p*gt).sum(1)/norm.clamp_min(1e-12)
        s[2]+=torch.acos(cos.clamp(-1,1))[mask].double().sum();pixels+=mask.sum()
    a=s.cpu().tolist()
    return dict(val_mse=a[0]/elements,val_psnr=a[1]/n,val_sam=math.degrees(a[2]/max(int(pixels),1)))

@torch.inference_mode()
def normalization_diagnostics(m, b):
    mean,scale=m.statistics(b[2]);normalized=[(x.float()-mean)/scale for x in b]
    roundtrip=max(float(((x*scale+mean)-y).abs().max()) for x,y in zip(normalized,b))
    p=m(*b);gain=1.1;offset=0.02
    q=m(*[x*gain+offset for x in b])
    error=(q-(p*gain+offset)).abs()
    return dict(roundtrip_max_abs=roundtrip,mean=mean.flatten().cpu().tolist(),scale=scale.flatten().cpu().tolist(),
                affine_gain=gain,affine_offset=offset,affine_output_max_abs=float(error.max()),affine_output_mean_abs=float(error.mean()),
                note='Affine equivariance is expected for N1 away from the std floor; quality is evaluated separately.')


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def setrng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda'])

def resume_probe(m,cache,cfg,arm):
    """Two tiny real-model AMP updates, checkpoint/reload between them, then rollback."""
    initial=cpu(m.state_dict());initial_rng=rng()
    folder=Path(cfg['out'])/'provenance';path=folder/f'resume_probe_{arm}.pt'
    optim=torch.optim.Adam(m.parameters(),lr=cfg['lr'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+9000)
    order=torch.randperm(4,device='cuda',generator=gen).cpu()
    def step(position):
        m.train();optim.zero_grad(set_to_none=True)
        ix=order[position*2:position*2+2].cuda();batch=[cache[k].index_select(0,ix) for k in KEYS]
        with torch.autocast('cuda',dtype=torch.float16):prediction=m(*batch[:3])
        loss=(prediction.float()-batch[3]).square().mean()
        scaler.scale(loss).backward();scaler.step(optim);scaler.update()
        return float(loss)
    try:
        step(0)
        atomic(path,dict(model=m.state_dict(),optimizer=optim.state_dict(),scaler=scaler.state_dict(),
                        rng=rng(),generator=gen.get_state(),order=order,next_batch=1))
        reference_loss=step(1);reference=cpu(m.state_dict());reference_optim=cpu(optim.state_dict())
        reference_scaler=copy.deepcopy(scaler.state_dict());reference_next=torch.randperm(4,device='cuda',generator=gen).cpu()
        ck=torch.load(path,map_location='cpu',weights_only=False)
        m.load_state_dict(ck['model'],strict=True)
        optim=torch.optim.Adam(m.parameters(),lr=cfg['lr']);optim.load_state_dict(ck['optimizer'])
        for state in optim.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.cuda()
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.);scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['generator']);order=ck['order'];setrng(ck['rng'])
        resumed_loss=step(ck['next_batch']);restored=cpu(m.state_dict())
        max_abs=0.
        for key,value in reference.items():
            other=restored[key]
            if value.is_floating_point():
                delta=float((value-other).abs().max());max_abs=max(max_abs,delta)
                torch.testing.assert_close(value,other,rtol=1e-5,atol=1e-6)
            elif not torch.equal(value,other):raise AssertionError('Resume integer buffer differs: '+key)
        def assert_tree(a,b):
            if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            elif isinstance(a,dict):
                if a.keys()!=b.keys():raise AssertionError('Resume optimizer structure differs')
                for k in a:assert_tree(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise AssertionError('Resume optimizer length differs')
                for x,y in zip(a,b):assert_tree(x,y)
            elif a!=b:raise AssertionError('Resume scalar differs')
        assert_tree(reference_optim,cpu(optim.state_dict()))
        assert_tree(reference_scaler,scaler.state_dict())
        if not torch.equal(reference_next,torch.randperm(4,device='cuda',generator=gen).cpu()):raise AssertionError('Generator restore differs')
        if abs(reference_loss-resumed_loss)>1e-6:raise AssertionError('Resume loss differs')
        return dict(passed=True,max_model_abs=max_abs,reference_loss=reference_loss,resumed_loss=resumed_loss,
                    scope='two real-model AMP updates on four train samples; optimizer/scaler/RNG/order/cursor restored',
                    limit='not a full epoch/session termination test; CUDA warn-only operations may not be bitwise deterministic')
    finally:
        m.load_state_dict(initial,strict=True);setrng(initial_rng);m.zero_grad(set_to_none=True)
        if path.exists():path.unlink()


def worker(cfg,arm):
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    out=Path(cfg['out'])/'results'/arm;out.mkdir(parents=True,exist_ok=True)
    checkpoints=Path(cfg['out'])/'checkpoints'/f'{arm}_QB_s{cfg["seed"]}';checkpoints.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+b')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def log(phase,**kw):
        rec=dict(arm=arm,gpu=cfg['gpu_map'][arm],phase=phase,time_kst=datetime.now(KST).isoformat(),allocated_gb=torch.cuda.memory_allocated()/2**30,reserved_gb=torch.cuda.memory_reserved()/2**30,peak_gb=torch.cuda.max_memory_allocated()/2**30,**kw)
        write(out/'status.json',rec)
        with (out/'events.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(rec,allow_nan=False)+'\n')
        print(json.dumps(rec,allow_nan=False),flush=True)
    m=Model(arm,cfg['std_floor']).cuda()
    initial_hash=digest_state(m.state_dict())
    write(out/'initialization.json',dict(state_sha256=initial_hash,parameters=sum(p.numel() for p in m.parameters()),
          initial_outputs_equal=False,initialization='fresh common seed; arm introduces no extra parameters'))
    m.eval()
    g=torch.Generator(device='cuda').manual_seed(123)
    test=[torch.rand(shape,device='cuda',generator=g) for shape in [(1,1,64,64),(1,4,64,64),(1,4,16,16)]]
    with torch.inference_mode():
        actual=m(*test)
        if arm=='N0':torch.testing.assert_close(actual,m.core(*test),rtol=0,atol=0)
        if not torch.isfinite(actual).all():raise FloatingPointError('Preflight forward')
        diagnostics=normalization_diagnostics(m,test)
        for value in (0.,0.5):
            constant=[torch.full_like(x,value) for x in test]
            if not torch.isfinite(m(*constant)).all():raise FloatingPointError('Constant-input forward')
    del actual,test;torch.cuda.empty_cache()
    optimizer=torch.optim.Adam(m.parameters(),lr=cfg['lr']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    updates=[0];optimizer.register_step_post_hook(lambda *a:updates.__setitem__(0,updates[0]+1));attempted=0
    log('loading_gpu_cache');train=gpu_cache(cfg['data']['train']);val=gpu_cache(cfg['data']['val'])
    # A backward preflight is rolled back completely before the actual run.
    probe_state=cpu(m.state_dict());probe_rng=rng();m.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):p=m(*[train[k][:cfg['micro']] for k in KEYS[:3]])
    loss=(p.float()-train['gt'][:cfg['micro']]).square().mean();scaler.scale(loss).backward()
    if not torch.isfinite(loss):raise FloatingPointError('Preflight loss')
    scaler.unscale_(optimizer)
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('Preflight gradients')
    m.load_state_dict(probe_state);setrng(probe_rng);optimizer=torch.optim.Adam(m.parameters(),lr=cfg['lr']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.);updates=[0];optimizer.register_step_post_hook(lambda *a:updates.__setitem__(0,updates[0]+1));optimizer.zero_grad(set_to_none=True)
    del probe_state,p,loss;torch.cuda.empty_cache()
    resume_check=resume_probe(m,train,cfg,arm)
    if digest_state(m.state_dict())!=initial_hash:raise ValueError('Preflight did not restore the common initial state')
    log('preflight_pass',initial_state_sha256=initial_hash,resume_probe=resume_check)
    write(out/'preflight.json',dict(forward=True,backward=True,gpu_cache=True,N0_original_equivalence=(arm=='N0'),normalization=diagnostics,micro=cfg['micro'],effective_batch=cfg['batch'],cuda_resume_equivalence=resume_check))
    if cfg['mode']=='check':return
    fingerprint=cfg['fingerprint'];latest=checkpoints/'latest.pt';history=[];epoch=0;next_batch=0;order=None;loss_sum=0.;elements=0;best_epoch=0;best_val=None;best_sd=None;durations=[];grad_diagnostics={}
    stop=[False];signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stop.__setitem__(0,True))
    generator=torch.Generator(device='cuda').manual_seed(seed+100000)
    if cfg['resume']:
        if not latest.exists():raise FileNotFoundError(f'Missing resume state: {latest}')
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['fingerprint']!=fingerprint:raise ValueError('Resume source/config/data mismatch; use a new run')
        m.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);generator.set_state(ck['generator'])
        
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.cuda()
        epoch=ck['epoch'];next_batch=ck['next_batch'];order=ck['order'];loss_sum=ck['loss_sum'];elements=ck['elements'];history=ck['history'];best_epoch=ck['best_epoch'];best_val=ck['best_val'];best_sd=ck['best_model'];attempted=ck['attempted'];updates[0]=ck['updates'];durations=ck['durations'];setrng(ck['rng'])
        atomic(checkpoints/'best.pt',dict(model=best_sd,best_epoch=best_epoch,metrics=best_val,fingerprint=fingerprint,config=cfg))
        write(out/'history.json',history)
    else:
        best_sd=cpu(m.state_dict())
        atomic(Path(cfg['out'])/'provenance'/f'initial_{arm}.pt',dict(model=best_sd,fingerprint=fingerprint,config=cfg))
        t=time.monotonic();best_val=validation(m,val,cfg['micro']);durations=[]
        history=[dict(epoch=0,train_mse=None,**best_val)]
        atomic(checkpoints/'best.pt',dict(model=best_sd,best_epoch=0,metrics=best_val,fingerprint=fingerprint,config=cfg));write(out/'history.json',history)
        log('epoch0_validation',**best_val)
    def save():
        atomic(latest,dict(model=m.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng(),generator=generator.get_state(),epoch=epoch,next_batch=next_batch,order=order,loss_sum=float(loss_sum),elements=elements,history=history,best_epoch=best_epoch,best_val=best_val,best_model=best_sd,attempted=attempted,updates=updates[0],durations=durations,fingerprint=fingerprint,config=cfg))
    save();last_save=time.monotonic();last_log=0.;n=len(train['pan']);steps=math.ceil(n/cfg['batch'])
    while epoch<cfg['epochs']:
        start=time.monotonic();start_batch=next_batch;m.train()
        if order is None:
            order=torch.randperm(n,device='cuda',generator=generator).cpu()
        order_gpu=order.cuda();order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        acc=torch.tensor(loss_sum,device='cuda',dtype=torch.float64)
        for bi in range(next_batch,steps):
            idx=order_gpu[bi*cfg['batch']:(bi+1)*cfg['batch']];denom=len(idx)*train['gt'][0].numel();optimizer.zero_grad(set_to_none=True)
            for j in range(0,len(idx),cfg['micro']):
                ix=idx[j:j+cfg['micro']];b=[train[k].index_select(0,ix) for k in KEYS]
                with torch.autocast('cuda',dtype=torch.float16):p=m(*b[:3])
                sq=(p.float()-b[3]).square().sum();scaler.scale(sq/denom).backward();acc+=sq.detach().double()
            if bi==steps-1:
                scaler.unscale_(optimizer)
                grad_diagnostics={name:float(torch.sqrt(sum((p.grad.float().square().sum() for p in module.parameters() if p.grad is not None),torch.zeros((),device='cuda')))) for name,module in [('core',m.core)]}
            attempted+=1;scaler.step(optimizer);scaler.update();elements+=denom;next_batch=bi+1
            now=time.monotonic()
            if now-last_log>=15 or bi==steps-1:
                loss_sum=float(acc)
                if not math.isfinite(loss_sum):raise FloatingPointError('Nonfinite train accumulation')
                eta=None
                if durations:
                    remaining=(cfg['epochs']-epoch-1)+(steps-next_batch)/steps
                    eta=float(np.mean(durations[-3:])*remaining)
                log('train',epoch=epoch+1,epochs=cfg['epochs'],batch=next_batch,batches=steps,train_mse=loss_sum/elements,lr=cfg['lr'],actual_updates=updates[0],attempted_updates=attempted,skipped_updates=attempted-updates[0],eta_seconds=eta,training_end_kst=(datetime.now(KST)+timedelta(seconds=eta)).isoformat() if eta is not None else '측정 대기',total_end_kst='평가·압축 I/O 미측정',eta_basis='recent full train+validation+checkpoint epochs; current pending validation/checkpoint omitted')
                last_log=now
            if now-last_save>=60 or stop[0]:
                loss_sum=float(acc);save();last_save=time.monotonic()
                if stop[0]:log('interrupted_saved');return
        loss_sum=float(acc);v=validation(m,val,cfg['micro']);epoch+=1
        row=dict(epoch=epoch,train_mse=loss_sum/elements,**v,order_sha256=order_hash,lr=cfg['lr'],updates=updates[0],attempted=attempted,skipped=attempted-updates[0],gradient_norm_last_batch=grad_diagnostics)
        if v['val_mse']<best_val['val_mse']:
            best_val=v;best_epoch=epoch;best_sd=cpu(m.state_dict())
        # best is always the true stored optimum, including epoch0.
        atomic(checkpoints/'best.pt',dict(model=best_sd,best_epoch=best_epoch,metrics=best_val,fingerprint=fingerprint,config=cfg))
        next_batch=0;order=None;loss_sum=0.;elements=0;history.append(row);write(out/'history.json',history)
        save();torch.cuda.synchronize();duration=time.monotonic()-start
        # Partial resumed epochs are excluded from ETA speed estimates.
        if start_batch==0:durations.append(duration)
        save();last_save=time.monotonic();log('epoch_complete',**row,epoch_seconds=duration,best_epoch=best_epoch)
    log('training_complete',best_epoch=best_epoch)
    if not cfg['evaluate_after_training']:
        write(out/'training_complete.json',dict(best_epoch=best_epoch,best_sha256=sha(checkpoints/'best.pt'),latest_sha256=sha(latest)))
        return
    evaluate(m,cfg,arm,out,log)
    write(out/'complete.json',dict(state='training_and_evaluation_complete',best_epoch=best_epoch,best_sha256=sha(checkpoints/'best.pt'),latest_sha256=sha(latest)))

def preview(arr,pred,path,label):
    from PIL import Image,ImageDraw
    bands=[2,1,0];ref=arr['gt'][bands].transpose(1,2,0);lo,hi=np.percentile(ref,[1,99],axis=(0,1))
    def rgb(a):return Image.fromarray((np.clip((a[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
    pn=arr['pan'][0];a,b=np.percentile(pn,[1,99]);pi=Image.fromarray((np.clip((pn-a)/max(b-a,1e-8),0,1)*255).astype('uint8')).convert('RGB');h,w=pn.shape
    canvas=Image.new('RGB',(4*w,h+32),'white');d=ImageDraw.Draw(canvas)
    for j,(txt,im) in enumerate([('LR MS',rgb(arr['ms'])),('PAN input',pi),(label,rgb(pred)),('GT',rgb(arr['gt']))]):canvas.paste(im.resize((w,h),Image.Resampling.NEAREST),(j*w,32));d.text((j*w+8,8),txt,fill='black')
    canvas.save(path)

@torch.inference_mode()
def evaluate(m,cfg,arm,out,log):
    checkpoints=Path(cfg['out'])/'checkpoints'/f'{arm}_QB_s{cfg["seed"]}'
    ck=torch.load(checkpoints/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval()
    records={};timing={};d=out/'results';d.mkdir(exist_ok=True)
    for proto in ('RR','FR'):
        rows=[];elapsed_scenes=[];log('evaluate_'+proto)
        with h5py.File(cfg['data'][proto],'r') as f:
            for i in range(len(f['pan'])):
                scene_start=time.monotonic()
                a={k:np.asarray(f[k][i],dtype=np.float64) for k in KEYS if k in f}
                inp=[torch.from_numpy((a[k]/2047).astype('float32')).unsqueeze(0).cuda() for k in KEYS[:3]]
                if i==0:
                    for _ in range(3):m(*inp)
                    times=[]
                    for _ in range(5):
                        torch.cuda.synchronize();start=time.perf_counter();m(*inp);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                    timing[proto]=dict(forward_median_ms=float(np.median(times)),repetitions=5,warmup=3,excludes='transfer and metrics')
                pred=m(*inp)[0].cpu().numpy().astype('float64')*2047
                if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite test prediction')
                metrics=rr_metrics(a['gt'],pred) if proto=='RR' else fr_metrics(pred,a['lms'],a['ms'],a['pan'])
                if not all(math.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite test metrics')
                rows.append(dict(index=i,metrics=metrics))
                if proto=='RR' and i in cfg['scenes']:preview(a,pred,d/f'RR_scene_{i:03}.png',arm)
                write(d/f'{proto}.json',dict(rows=rows,mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in metrics}))
                elapsed_scenes.append(time.monotonic()-scene_start)
                remaining=float(np.mean(elapsed_scenes[-5:])*(len(f['pan'])-i-1))
                log('evaluate_'+proto,scene=i+1,scenes=len(f['pan']),protocol_eta_seconds=remaining,protocol_end_kst=(datetime.now(KST)+timedelta(seconds=remaining)).isoformat(),total_end_kst='Remaining protocol or archive I/O not yet measured')
        records[proto]=dict(rows=rows,mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']})
    write(d/'metrics.json',dict(protocols=records,best_epoch=ck['best_epoch'],timing=timing,limits=['QNR/Q2n/SCC MATLAB parity unverified','Previously inspected benchmark; no new independent holdout','Single QB seed; fresh-from-scratch comparison']))

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text());arm=sys.argv[2]
    try:worker(cfg,arm)
    except BaseException:
        out=Path(cfg['out'])/'results'/arm;out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),time=time.time()));raise

