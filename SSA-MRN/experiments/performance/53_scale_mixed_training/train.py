import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,math,random,hashlib,traceback,signal,platform
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
import torch
import h5py
sys.path.append(str(Path(__file__).resolve().parents[3]))  # repository common/ (interpolation, observation) after the 43 source copy
from model import Model,ARMS,Parallel
from prepare import CoarsePairs
from common.metrics import rr_metrics,fr_metrics
HERE=Path(__file__).resolve().parent
KEYS=('pan','lms','ms','gt');KST=timezone(timedelta(hours=9))
def now():return datetime.now(KST).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def replace(src,dest):
    for attempt in range(40):
        try:os.replace(src,dest);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');replace(t,p)
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x
def state_hash(sd):
    h=hashlib.sha256()
    for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(cpu(obj),tmp)
    with tmp.open('r+b') as f:f.flush();os.fsync(f.fileno())
    replace(tmp,p)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def setrng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda'])
def setup(seed):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def lock(p):
    f=Path(p).open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
    if os.name=='nt':
        import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f

def cache(cfg,split):return {k:np.load(Path(cfg.get('cache_root') or Path(cfg['runtime'])/'cache')/f'{split}_{k}.npy',mmap_mode='r') for k in KEYS}
def coarse_cache(cfg,split):return {k:np.load(Path(cfg['runtime'])/'cache'/f'coarse_{split}_{k}.npy',mmap_mode='r') for k in KEYS}
def batch(c,idx,layout):
    b=[]
    for k in KEYS:
        x=torch.from_numpy(np.array(c[k][idx],copy=True,order='C')).pin_memory().to('cuda',non_blocking=True)
        if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last)
        b.append(x)
    return b
@torch.inference_mode()
def validation(m,c,cfg):
    m.eval();acc=torch.zeros(3,device='cuda',dtype=torch.float64);pixels=torch.zeros((),device='cuda',dtype=torch.int64);elements=0;n=0
    for i in range(0,len(c['pan']),cfg['micro']):
        b=batch(c,slice(i,i+cfg['micro']),cfg['layout']);p=m(*b[:3]).float();gt=b[3]
        if not torch.isfinite(p).all():raise FloatingPointError('nonfinite validation')
        err=(p-gt).square();acc[0]+=err.double().sum();elements+=gt.numel();n+=len(gt)
        acc[1]+=(-10*torch.log10(err.mean((2,3)).clamp_min(1e-15))).mean(1).double().sum()
        norm=p.norm(dim=1)*gt.norm(dim=1);mask=norm>1e-12;cos=(p*gt).sum(1)/norm.clamp_min(1e-12)
        acc[2]+=torch.acos(cos.clamp(-1,1))[mask].double().sum();pixels+=mask.sum()
    a=acc.cpu().tolist();return dict(val_mse=a[0]/elements,val_psnr=a[1]/n,val_sam=math.degrees(a[2]/max(int(pixels),1)))
def make_model(arm,seed,layout):
    m=Model(arm,seed).cuda()
    if layout=='channels_last':m=m.to(memory_format=torch.channels_last)
    return m

def benchmark(cfg,arm,layout,tag='single'):
    # Disposable throughput probe with the real per-step path (mmap cache -> pinned batch -> AMP step); not a trained checkpoint.
    setup(42);cfg={**cfg,'layout':layout};m=make_model(arm,42,layout);c=cache(cfg,'train');cc=coarse_cache(cfg,'train');n=len(c['pan'])
    gen=torch.Generator().manual_seed(7);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr_start']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    m.train();samples=[];steps=0
    for i in range(84):
        torch.cuda.synchronize();t=time.perf_counter();idx=torch.randperm(n,generator=gen)[:cfg['batch']].sort().values.numpy()
        b=batch(c,idx,layout);bc=batch(cc,idx,layout);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):p=m(*b[:3]);pc=m(*bc[:3])
        loss=(p.float()-b[3]).square().mean()+cfg['coarse_weight']*(pc.float()-bc[3]).square().mean();scaler.scale(loss).backward();scaler.unscale_(opt)
        if i==0 and (not torch.isfinite(loss) or not all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())):raise FloatingPointError('benchmark nonfinite')
        scaler.step(opt);scaler.update();torch.cuda.synchronize()
        if i>=4:samples.append(time.perf_counter()-t)
        steps+=1
    result=dict(arm=arm,layout=layout,tag=tag,median_step_seconds=float(np.median(samples)),measured_steps=len(samples),warmup=4,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss=float(loss),probe_updates=steps,time_kst=now(),note='disposable benchmark model; not an experimental trained checkpoint')
    write(Path(cfg['runtime'])/'preflight'/f'benchmark_{tag}_{arm}.json',result);print(json.dumps(result),flush=True)

def preflight(cfg):
    setup(42);out=Path(cfg['runtime'])/'preflight';out.mkdir(exist_ok=True)
    import model as M52
    import prepare as C52
    from common.observation import mtf_kernels
    M48=M52.M48
    # 1. Initial weights equal the 48/43/49 controls of the same seed (same constructor and RNG path).
    core_hashes={}
    for seed in cfg['seeds']:
        for arm in ARMS:
            h=state_hash(Model(arm,seed).state_dict());core_hashes[f'{arm}_s{seed}']=h
            if h!=state_hash(M48.Model(M52.BASE[arm],seed).state_dict()):raise AssertionError('initial state differs from 48 '+arm)
            core={k:v for k,v in Model(arm,seed).state_dict().items() if k.startswith('core.')}
            expected=cfg['expected_43_initial_state_sha256'].get(f"{M48.CORE[M52.BASE[arm]]}_s{seed}")
            if expected and state_hash(core)!=expected:raise AssertionError('core differs from recorded 43 initial state '+arm)
    # 2. Coarse-pair operator: the x2 genMTF construction reproduces the audited x4 kernels when ratio=4.
    ref=mtf_kernels('QB');mine=np.stack([C52.gen_mtf(g,ratio=4) for g in C52.GNYQ['QB']])[:,None]
    kernel_diff=float(np.abs(ref-mine).max())
    if kernel_diff>1e-12:raise AssertionError('genMTF reconstruction differs from common/observation')
    cp=C52.CoarsePairs('cuda');ms_dc=[float(k.sum()) for k in cp.ms_kernel];pan_dc=float(cp.pan_kernel.sum())
    # 3. Coarse cache equals a fresh computation, shapes and consistency with the given MS (2x reduction of MS ~ MS').
    tr=cache(cfg,'train');trc=coarse_cache(cfg,'train');idx=np.arange(0,len(tr['pan']),997)[:16]
    b=[torch.from_numpy(np.array(tr[k][idx])).cuda() for k in KEYS];bc=[torch.from_numpy(np.array(trc[k][idx])).cuda() for k in KEYS]
    fresh=cp(b[0],b[3]);cache_diff=max(float((x-y).abs().max()) for x,y in zip(fresh,bc))
    if cache_diff>1e-6:raise AssertionError('coarse cache differs from a fresh computation')
    shapes={k:list(v.shape) for k,v in trc.items()}
    ms_half=cp._blur_down(b[2].double(),cp.ms_kernel).float()
    consistency=float(((ms_half-bc[2]).square().mean().sqrt()/b[2].square().mean().sqrt()))
    # 4. Forward/backward on both scales, AMP two-update resume, rollback (as 43/48).
    torch.manual_seed(991);inputs=[torch.rand(s,device='cuda') for s in [(2,1,64,64),(2,4,64,64),(2,4,16,16)]];coarse_in=[torch.rand(s,device='cuda') for s in [(2,1,32,32),(2,4,32,32),(2,4,8,8)]]
    if cfg['layout']=='channels_last':inputs=[x.contiguous(memory_format=torch.channels_last) for x in inputs];coarse_in=[x.contiguous(memory_format=torch.channels_last) for x in coarse_in]
    models={a:make_model(a,42,cfg['layout']).eval() for a in ARMS};checks={}
    with torch.inference_mode():
        for a,m in models.items():
            if not (torch.isfinite(m(*inputs)).all() and torch.isfinite(m(*coarse_in)).all()):raise FloatingPointError('initial forward '+a)
    for arm,m in models.items():
        before=state_hash(m.state_dict());m.train();initial=cpu(m.state_dict());initial_rng=rng()
        opt=torch.optim.Adam(m.parameters(),lr=cfg['lr_start']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):p=m(*inputs);pc=m(*coarse_in)
            loss=(p.float()-inputs[1]).square().mean()+cfg['coarse_weight']*(pc.float()-coarse_in[1]).square().mean();scaler.scale(loss).backward();scaler.unscale_(opt)
            if not torch.isfinite(loss) or not all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters()):raise FloatingPointError('preflight gradients')
            scaler.step(opt);scaler.update();return float(loss)
        step();checkpoint=dict(model=cpu(m.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),rng=rng())
        expected_loss=step();expected=cpu(m.state_dict())
        m.load_state_dict(checkpoint['model']);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr_start']);opt.load_state_dict(checkpoint['optimizer'])
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.);scaler.load_state_dict(checkpoint['scaler']);setrng(checkpoint['rng'])
        actual_loss=step();largest=max(float((v-m.state_dict()[k].cpu()).abs().max()) for k,v in expected.items())
        if abs(expected_loss-actual_loss)>1e-6 or largest>1e-6:raise AssertionError('resume differs')
        m.load_state_dict(initial);setrng(initial_rng)
        if state_hash(m.state_dict())!=before:raise AssertionError('probe rollback')
        print(json.dumps(dict(preflight_arm=arm,resume_max_abs=largest)),flush=True)
        checks[arm]=dict(initial_state_sha256=before,parameters=sum(p.numel() for p in m.parameters()),resume_max_abs=largest,loss=actual_loss)
    write(out/'checks.json',dict(time_kst=now(),initial_state_equal_to_48=True,initial_state_sha256=core_hashes,genmtf_x4_max_abs_diff=kernel_diff,
        coarse_kernel_dc_gain=dict(ms=ms_dc,pan=pan_dc),coarse_cache_vs_fresh_max_abs=cache_diff,coarse_shapes=shapes,
        coarse_ms_vs_2x_reduced_given_ms_relative_rmse=consistency,checks=checks,scope='tiny AMP forward/backward on both scales and two-update resume; not long-session crash proof'))

def train(cfg,arm,seed,resume=False):
    setup(seed);root=Path(cfg['runtime']);out=root/'runs'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock');cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    def log(phase,**kw):
        rec=dict(time_kst=now(),pid=os.getpid(),arm=arm,seed=seed,phase=phase,allocated_gb=torch.cuda.memory_allocated()/2**30,reserved_gb=torch.cuda.memory_reserved()/2**30,**kw)
        write(out/'status.json',rec)
        with (out/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)
    if (out/'training_complete.json').exists():return
    if (cp/'initial.pt').exists() and not resume:raise RuntimeError('existing run; explicit --resume required')
    log('initializing');m=make_model(arm,seed,cfg['layout']);initial_hash=state_hash(m.state_dict());opt=torch.optim.Adam(m.parameters(),lr=cfg['lr_start']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    updates=[0];opt.register_step_post_hook(lambda *a:updates.__setitem__(0,updates[0]+1));attempted=0
    tr=cache(cfg,'train');va=cache(cfg,'val');trc=coarse_cache(cfg,'train');vac=coarse_cache(cfg,'val');n=len(tr['pan']);steps=math.ceil(n/cfg['batch']);cw=cfg['coarse_weight'];csum=0.;celem=0
    gen=torch.Generator().manual_seed(seed+100000);epoch=0;cursor=0;order=None;loss_sum=0.;elements=0;history=[];durations=[];best=None;best_epoch=0;best_sd=None
    stop=[False];signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stop.__setitem__(0,True))
    if resume and (cp/'latest.pt').exists():
        ck=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
        if ck['fingerprint']!=cfg['fingerprint']:
            if ck['fingerprint'] not in cfg.get('compatible_resume_fingerprints',[]):raise RuntimeError('resume contract mismatch')
            scientific=('data_sha256','cache_manifest_sha256','seeds','epochs','batch','micro','lr_start','lr_end','optimizer','K','spatial_ratio','std_floor','precision','tf32','cudnn_benchmark','cudnn_deterministic','torch_deterministic_algorithms','cublas_workspace_config','layout','queue','loss','selection')
            for key in scientific:
                if ck['config'][key]!=cfg[key]:raise RuntimeError('scientific condition changed on operational resume: '+key)
            for path,digest in ck['config']['source_hashes'].items():
                if path not in ('train.py','run.py') and cfg['source_hashes'][path]!=digest:raise RuntimeError('model/data source changed on operational resume: '+path)
        m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);setrng(ck['rng']);gen.set_state(ck['generator'])
        epoch=ck['epoch'];cursor=ck['next_batch'];order=ck['order'];loss_sum=ck['loss_sum'];elements=ck['elements'];csum=ck.get('coarse_loss_sum',0.);celem=ck.get('coarse_elements',0);history=ck['history'];durations=ck['durations'];best=ck['best_val'];best_epoch=ck['best_epoch'];best_sd=ck['best_model'];attempted=ck['attempted'];updates[0]=ck['updates']
    elif resume and (cp/'initial.pt').exists():raise RuntimeError('initial exists without latest; recovery required')
    else:
        best_sd=cpu(m.state_dict());atomic(cp/'initial.pt',dict(model=best_sd,config=cfg,arm=arm,seed=seed,initial_state_sha256=initial_hash))
        best=validation(m,va,cfg);vc0=validation(m,vac,cfg);history=[dict(epoch=0,train_mse=None,**best,**{'coarse_'+k:v for k,v in vc0.items()})]
        atomic(cp/'best.pt',dict(model=best_sd,arm=arm,seed=seed,best_epoch=0,metrics=best,config=cfg,fingerprint=cfg['fingerprint']))
        write(out/'initialization.json',dict(state_sha256=initial_hash,parameters=sum(p.numel() for p in m.parameters()),seed=seed));log('epoch0_validation',**best)
    def save():
        atomic(cp/'latest.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng(),generator=gen.get_state(),epoch=epoch,next_batch=cursor,order=order,loss_sum=loss_sum,elements=elements,coarse_loss_sum=csum,coarse_elements=celem,history=history,durations=durations,best_val=best,best_epoch=best_epoch,best_model=best_sd,attempted=attempted,updates=updates[0],arm=arm,seed=seed,config=cfg,fingerprint=cfg['fingerprint']))
    save();last_save=time.monotonic();last_log=0.
    while epoch<cfg['epochs']:
        start=time.monotonic();start_cursor=cursor;m.train()
        lr=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*epoch/(cfg['epochs']-1)))
        for group in opt.param_groups:group['lr']=lr
        if order is None:order=torch.randperm(n,generator=gen)
        order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest();acc=torch.tensor(loss_sum,device='cuda',dtype=torch.float64);accc=torch.tensor(csum,device='cuda',dtype=torch.float64)
        for bi in range(cursor,steps):
            idx=order[bi*cfg['batch']:(bi+1)*cfg['batch']].numpy();b=batch(tr,idx,cfg['layout']);opt.zero_grad(set_to_none=True)
            bc=batch(trc,idx,cfg['layout'])
            with torch.autocast('cuda',dtype=torch.float16):p=m(*b[:3]);pc=m(*bc[:3])
            sq=(p.float()-b[3]).square().sum();sqc=(pc.float()-bc[3]).square().sum()
            # Same patches at the original and the 2x coarser scale; per-pixel means summed with weight cw.
            scaler.scale(sq/b[3].numel()+cw*sqc/bc[3].numel()).backward();acc+=sq.detach().double();accc+=sqc.detach().double();elements+=b[3].numel();celem+=bc[3].numel();attempted+=1
            scaler.step(opt);scaler.update();cursor=bi+1;t=time.monotonic()
            if t-last_log>=15 or bi==steps-1:
                loss_sum=float(acc);csum=float(accc)
                if not math.isfinite(loss_sum):raise FloatingPointError('nonfinite train accumulation')
                recent=float(np.median(durations[-3:])) if durations else None
                eta=recent*(cfg['epochs']-epoch-1+(steps-cursor)/steps) if recent else None
                log('train',epoch=epoch+1,epochs=cfg['epochs'],batch=cursor,batches=steps,train_mse=loss_sum/elements,lr=lr,actual_updates=updates[0],skipped_updates=attempted-updates[0],epoch_seconds_estimate=recent,eta_seconds=eta,training_end_kst=(datetime.now(KST)+timedelta(seconds=eta)).isoformat() if eta else None)
                last_log=t
            if t-last_save>=60 or stop[0]:
                loss_sum=float(acc);csum=float(accc);save();last_save=time.monotonic()
                if stop[0]:log('interrupted_saved');return
        loss_sum=float(acc);csum=float(accc);v=validation(m,va,cfg);vc=validation(m,vac,cfg);epoch+=1
        row=dict(epoch=epoch,train_mse=loss_sum/elements,coarse_train_mse=csum/max(celem,1),**v,**{'coarse_'+k:x for k,x in vc.items()},lr=lr,order_sha256=order_hash,updates=updates[0],attempted=attempted,skipped=attempted-updates[0])
        if v['val_mse']<best['val_mse']:
            best=v;best_epoch=epoch;best_sd=cpu(m.state_dict());atomic(cp/'best.pt',dict(model=best_sd,arm=arm,seed=seed,best_epoch=epoch,metrics=best,config=cfg,fingerprint=cfg['fingerprint']))
        cursor=0;order=None;loss_sum=0.;elements=0;csum=0.;celem=0;history.append(row);write(out/'history.json',history);save();torch.cuda.synchronize();duration=time.monotonic()-start
        if start_cursor==0:durations.append(duration)
        history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();log('epoch_complete',**row,best_epoch=best_epoch)
    expected_updates=cfg['epochs']*steps
    if attempted!=expected_updates:raise AssertionError('sample/update count mismatch')
    write(out/'training_complete.json',dict(time_kst=now(),epoch=epoch,best_epoch=best_epoch,best_val=best,best_sha256=sha(cp/'best.pt'),latest_sha256=sha(cp/'latest.pt'),attempted=attempted,updates=updates[0],fingerprint=cfg['fingerprint']))
    log('training_complete',best_epoch=best_epoch)

@torch.inference_mode()
def evaluate(cfg,arm,seed):
    setup(seed);out=Path(cfg['runtime'])/'runs'/f'{arm}_s{seed}';done=out/'evaluation_complete.json'
    if done.exists():return
    m=make_model(arm,seed,cfg['layout']);p=out/'checkpoints/best.pt';ck=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);m.eval()
    dest=out/'evaluation';dest.mkdir(exist_ok=True);weight_sha=sha(p);records={}
    from PIL import Image,ImageDraw
    def preview(a,pred,path):
        bands=[2,1,0];ref=a['gt'][bands].transpose(1,2,0);lo,hi=np.percentile(ref,[1,99],axis=(0,1))
        def rgb(x):return Image.fromarray((np.clip((x[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
        pan=a['pan'][0];pl,ph=np.percentile(pan,[1,99]);pi=Image.fromarray((np.clip((pan-pl)/max(ph-pl,1e-8),0,1)*255).astype('uint8')).convert('RGB');h,w=pan.shape
        image=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(image)
        for j,(txt,im) in enumerate([('LR MS',rgb(a['ms'])),('PAN input',pi),(arm,rgb(pred)),('GT',rgb(a['gt']))]):image.paste(im.resize((w,h),Image.Resampling.NEAREST),(j*w,32));draw.text((j*w+8,8),txt,fill='black')
        image.save(path)
    for proto in ('RR','FR'):
        rows=[]
        with h5py.File(Path(cfg['data_root'])/cfg['data'][proto]) as f:
            for i in range(len(f['pan'])):
                a={k:np.asarray(f[k][i],dtype=np.float64) for k in KEYS if k in f};b=[torch.from_numpy((a[k]/2047).astype('float32')).unsqueeze(0).cuda() for k in KEYS[:3]]
                if cfg['layout']=='channels_last':b=[x.contiguous(memory_format=torch.channels_last) for x in b]
                pred=m(*b)[0].cpu().numpy().astype('float64')*2047
                if not np.isfinite(pred).all():raise FloatingPointError('test nonfinite')
                metrics=rr_metrics(a['gt'],pred) if proto=='RR' else fr_metrics(pred,a['lms'],a['ms'],a['pan'])
                rows.append(dict(index=i,metrics=metrics))
                if proto=='RR' and i in cfg['scenes']:preview(a,pred,dest/f'RR_scene_{i:03}.png')
                result=dict(best_epoch=ck['best_epoch'],weight_sha256=weight_sha,rows=rows,mean={k:float(np.mean([x['metrics'][k] for x in rows])) for k in metrics})
                write(dest/f'{proto}.json',result);print(json.dumps(dict(arm=arm,seed=seed,phase='evaluate_'+proto,scene=i+1,time_kst=now())),flush=True)
        if len(rows)!=20:raise AssertionError('expected full20')
        records[proto]=result
    write(done,dict(time_kst=now(),best_epoch=ck['best_epoch'],weight_sha256=weight_sha,means={p:x['mean'] for p,x in records.items()},limits=cfg['limits']))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'runtime_config.json'));p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42);p.add_argument('--layout',default='contiguous');p.add_argument('--resume',action='store_true');p.add_argument('--tag',default='single');a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    try:
        if a.mode=='bench':benchmark(cfg,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':preflight(cfg)
        elif a.mode=='evaluate':evaluate(cfg,a.arm,a.seed)
        else:train(cfg,a.arm,a.seed,a.resume)
    except BaseException:
        folder=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else f'runs/{a.arm}_s{a.seed}');folder.mkdir(parents=True,exist_ok=True);write(folder/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
