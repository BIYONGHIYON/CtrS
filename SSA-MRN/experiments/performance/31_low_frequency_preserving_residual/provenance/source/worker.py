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
from refiner import RefinementNet
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

class Model(RefinementNet):
    def __init__(self,arm,cal):
        super().__init__('G5',4);self.configure(cal);self.arm=arm;self.capture=False;self.diagnostics=[]
        t=torch.arange(-4,5,dtype=torch.float32);v=torch.exp(-t.square()/8);v/=v.sum()
        self.register_buffer('hp_kernel',(v[:,None]*v[None,:])[None,None].repeat(4,1,1,1),persistent=False)
    def low(self,c):return F.conv2d(F.pad(c,(4,4,4,4),mode='reflect'),self.hp_kernel.to(c.dtype),groups=4)
    def forward(self,pan,lms,ms):
        x=self.core(pan,lms,ms);self.diagnostics=[]
        for i in range(2):
            em=F.interpolate(ms-self.down(x,ms.shape[-2:]),size=x.shape[-2:],mode='bilinear',align_corners=True)
            ep=pan-(x*self.pan_weights).sum(1,keepdim=True)-self.pan_bias
            ep=ep-F.avg_pool2d(F.pad(ep,(2,2,2,2),mode='reflect'),5,stride=1)
            c=self.refiners[i](torch.cat((x,lms,pan,em,ep),1))
            if self.arm=='P1' or self.capture:lo=self.low(c)
            used=c-lo if self.arm=='P1' else c
            if self.capture:
                self.diagnostics.append(dict(raw_rms=float(c.float().square().mean().sqrt()),low_rms=float(lo.float().square().mean().sqrt()),used_rms=float(used.float().square().mean().sqrt()),band_mean=used.float().mean((0,2,3)).detach().cpu().tolist()))
            x=x+used
        return x

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
    m.eval();s=torch.zeros(3,device='cuda',dtype=torch.float64);pixels=torch.zeros((),device='cuda',dtype=torch.int64);n=0;elements=0;diag=None
    for i in range(0,len(cache['pan']),micro):
        b=[cache[k][i:i+micro] for k in KEYS];gt=b[3];m.capture=(i==0)
        p=m(*b[:3]).float()
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite validation prediction')
        if i==0:
            diag=copy.deepcopy(m.diagnostics)
            obs=float((m.down(p,b[2].shape[-2:])-b[2]).square().mean())
        s[0]+=(p-gt).square().sum().double();elements+=gt.numel()
        s[1]+=(-10*torch.log10((p-gt).square().mean((2,3)).clamp_min(1e-15))).mean(1).sum().double();n+=len(gt)
        norm=p.norm(dim=1)*gt.norm(dim=1);mask=norm>1e-12
        cos=(p*gt).sum(1)/norm.clamp_min(1e-12)
        s[2]+=torch.acos(cos.clamp(-1,1))[mask].sum().double();pixels+=mask.sum()
    m.capture=False
    a=s.cpu().tolist();return dict(val_mse=a[0]/elements,val_psnr=a[1]/n,val_sam=math.degrees(a[2]/max(int(pixels),1)),diagnostics_first_batch=diag,ms_observation_mse_first_batch=obs)

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def setrng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda'])

def worker(cfg,arm):
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    out=Path(cfg['out'])/arm;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+b')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def log(phase,**kw):
        rec=dict(arm=arm,gpu=cfg['gpu_map'][arm],phase=phase,time_kst=datetime.now(KST).isoformat(),allocated_gb=torch.cuda.memory_allocated()/2**30,reserved_gb=torch.cuda.memory_reserved()/2**30,peak_gb=torch.cuda.max_memory_allocated()/2**30,**kw)
        write(out/'status.json',rec);print(json.dumps(rec,allow_nan=False),flush=True)
    parent=torch.load(cfg['parent'],map_location='cpu',weights_only=False)
    if sha(cfg['parent'])!=cfg['parent_sha256']:raise ValueError('Parent changed')
    m=Model(arm,cfg['calibration']);m.load_state_dict(parent['model'],strict=True);m=m.cuda()
    initial_hash=digest_state(m.state_dict());write(out/'initialization.json',dict(state_sha256=initial_hash,parent_sha256=cfg['parent_sha256'],parent_best_epoch=parent.get('best_epoch'),parameters=sum(p.numel() for p in m.parameters())))
    # Verify the P0 implementation against the unmodified retained G5 forward.
    old=RefinementNet('G5',4);old.configure(cfg['calibration']);old.load_state_dict(parent['model'],strict=True);old=old.cuda().eval();m.eval()
    g=torch.Generator(device='cuda').manual_seed(123)
    test=[torch.rand(s,device='cuda',generator=g) for s in [(1,1,64,64),(1,4,64,64),(1,4,16,16)]]
    with torch.inference_mode():
        actual=m(*test);expected=old(*test)
        if arm=='P0':torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if not torch.isfinite(actual).all():raise FloatingPointError('Preflight forward')
    del old,actual,expected,test;torch.cuda.empty_cache()
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
    del probe_state,p,loss;torch.cuda.empty_cache();log('preflight_pass',initial_state_sha256=initial_hash)
    write(out/'preflight.json',dict(forward=True,backward=True,gpu_cache=True,P0_original_equivalence=(arm=='P0'),micro=cfg['micro'],effective_batch=cfg['batch']))
    if cfg['mode']=='check':return
    fingerprint=cfg['fingerprint'];latest=out/'latest.pt';history=[];epoch=0;next_batch=0;order=None;loss_sum=0.;elements=0;best_epoch=0;best_val=None;best_sd=None;durations=[];grad_diagnostics={}
    stop=[False];signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stop.__setitem__(0,True))
    generator=torch.Generator(device='cuda').manual_seed(seed+100000)
    if cfg['resume']:
        if not latest.exists():raise FileNotFoundError(f'Missing resume state: {latest}')
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['fingerprint']!=fingerprint:raise ValueError('Resume source/config/data mismatch; use a new run')
        m.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);generator.set_state(ck['generator'])
        epoch=ck['epoch'];next_batch=ck['next_batch'];order=ck['order'];loss_sum=ck['loss_sum'];elements=ck['elements'];history=ck['history'];best_epoch=ck['best_epoch'];best_val=ck['best_val'];best_sd=ck['best_model'];attempted=ck['attempted'];updates[0]=ck['updates'];durations=ck['durations'];setrng(ck['rng'])
        if not (out/'best.pt').exists():atomic(out/'best.pt',dict(model=best_sd,best_epoch=best_epoch,metrics=best_val,fingerprint=fingerprint,config=cfg))
        write(out/'history.json',history)
    else:
        t=time.monotonic();best_val=validation(m,val,cfg['micro']);best_sd=cpu(m.state_dict());durations=[]
        history=[dict(epoch=0,train_mse=None,**best_val)]
        atomic(out/'best.pt',dict(model=best_sd,best_epoch=0,metrics=best_val,fingerprint=fingerprint,config=cfg));write(out/'history.json',history)
        log('epoch0_validation',**best_val)
    def save():
        atomic(latest,dict(model=m.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng(),generator=generator.get_state(),epoch=epoch,next_batch=next_batch,order=order,loss_sum=float(loss_sum),elements=elements,history=history,best_epoch=best_epoch,best_val=best_val,best_model=best_sd,attempted=attempted,updates=updates[0],durations=durations,fingerprint=fingerprint,config=cfg))
    save();last_save=time.monotonic();last_log=0.;n=len(train['pan']);steps=math.ceil(n/cfg['batch'])
    while epoch<cfg['epochs']:
        start=time.monotonic();start_batch=next_batch;m.train();m.capture=False
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
                grad_diagnostics={name:float(torch.sqrt(sum((p.grad.float().square().sum() for p in module.parameters() if p.grad is not None),torch.zeros((),device='cuda')))) for name,module in [('core',m.core),('refiner0',m.refiners[0]),('refiner1',m.refiners[1])]}
            attempted+=1;scaler.step(optimizer);scaler.update();elements+=denom;next_batch=bi+1
            now=time.monotonic()
            if now-last_log>=15 or bi==steps-1:
                loss_sum=float(acc)
                if not math.isfinite(loss_sum):raise FloatingPointError('Nonfinite train accumulation')
                eta=None
                if durations:
                    remaining=(cfg['epochs']-epoch-1)+(steps-next_batch)/steps
                    eta=float(np.mean(durations[-3:])*remaining)
                log('train',epoch=epoch+1,epochs=cfg['epochs'],batch=next_batch,batches=steps,train_mse=loss_sum/elements,lr=cfg['lr'],actual_updates=updates[0],attempted_updates=attempted,skipped_updates=attempted-updates[0],eta_seconds=eta,training_end_kst=(datetime.now(KST)+timedelta(seconds=eta)).isoformat() if eta is not None else '측정 대기',total_end_kst='평가·압축 I/O 미측정',eta_basis='recent measured complete epochs; current validation/checkpoint not included')
                last_log=now
            if now-last_save>=60 or stop[0]:
                loss_sum=float(acc);save();last_save=time.monotonic()
                if stop[0]:log('interrupted_saved');return
        loss_sum=float(acc);v=validation(m,val,cfg['micro']);epoch+=1
        row=dict(epoch=epoch,train_mse=loss_sum/elements,**v,order_sha256=order_hash,lr=cfg['lr'],updates=updates[0],attempted=attempted,skipped=attempted-updates[0],gradient_norm_last_batch=grad_diagnostics)
        if v['val_mse']<best_val['val_mse']:
            best_val=v;best_epoch=epoch;best_sd=cpu(m.state_dict())
        # best is always the true stored optimum, including epoch0.
        atomic(out/'best.pt',dict(model=best_sd,best_epoch=best_epoch,metrics=best_val,fingerprint=fingerprint,config=cfg))
        next_batch=0;order=None;loss_sum=0.;elements=0;history.append(row);write(out/'history.json',history)
        save();torch.cuda.synchronize();duration=time.monotonic()-start
        # Partial resumed epochs are excluded from ETA speed estimates.
        if start_batch==0:durations.append(duration)
        save();last_save=time.monotonic();log('epoch_complete',**row,epoch_seconds=duration,best_epoch=best_epoch)
    log('training_complete',best_epoch=best_epoch)
    evaluate(m,cfg,arm,out,log)
    write(out/'complete.json',dict(state='training_and_evaluation_complete',best_epoch=best_epoch,best_sha256=sha(out/'best.pt'),latest_sha256=sha(latest)))

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
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval();m.capture=False
    records={};timing={};d=out/'results';d.mkdir(exist_ok=True)
    for proto in ('RR','FR'):
        rows=[];log('evaluate_'+proto)
        with h5py.File(cfg['data'][proto],'r') as f:
            for i in range(len(f['pan'])):
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
                log('evaluate_'+proto,scene=i+1,scenes=len(f['pan']))
        records[proto]=dict(rows=rows,mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']})
    write(d/'metrics.json',dict(protocols=records,best_epoch=ck['best_epoch'],timing=timing,limits=['QNR/Q2n/SCC MATLAB parity unverified','Repeated exploratory test; no new independent holdout','Single QB seed46; pretrained G5-conditioned comparison']))

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text());arm=sys.argv[2]
    try:worker(cfg,arm)
    except BaseException:
        out=Path(cfg['out'])/arm;out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
