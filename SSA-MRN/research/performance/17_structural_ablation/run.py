"""Detached two-slot Windows QB structure queue, AMP, mmap, atomic mid-epoch recovery."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, contextlib, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
StructureNet = __import__('importlib').import_module('research.performance.17_structural_ablation.model').StructureNet; VARIANTS = __import__('importlib').import_module('research.performance.17_structural_ablation.model').VARIANTS
KEYS = ('pan', 'lms', 'ms', 'gt')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8'); os.replace(t, p)

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=True

@contextlib.contextmanager
def lock(path):
    f = Path(path).open('a+b'); f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        f.close()

class Cached(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder); self.arrays = None
        self.n = len(np.load(self.folder/'pan.npy', mmap_mode='r'))
    def __len__(self): return self.n
    def __getitem__(self, i):
        if self.arrays is None: self.arrays = {k: np.load(self.folder/(k+'.npy'), mmap_mode='r') for k in KEYS}
        return tuple(torch.from_numpy(np.array(self.arrays[k][i], copy=True)) for k in KEYS)
    def __getstate__(self): return dict(folder=self.folder, n=self.n, arrays=None)

class Remaining:
    def __init__(self, n, epoch, start, batch=32): self.n=n; self.epoch=epoch; self.start=start; self.batch=batch
    def __iter__(self):
        order=torch.randperm(self.n, generator=torch.Generator().manual_seed(42+self.epoch)).tolist()
        for i in range(self.start*self.batch, self.n, self.batch): yield order[i:i+self.batch]
    def __len__(self): return math.ceil(self.n/self.batch)-self.start

def loader(ds, workers, sampler=None):
    opts=dict(num_workers=workers, pin_memory=True)
    if workers: opts.update(persistent_workers=True, prefetch_factor=1)
    if sampler is not None: return DataLoader(ds, batch_sampler=sampler, **opts)
    return DataLoader(ds, batch_size=32, shuffle=False, **opts)

def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    if isinstance(x, dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return x

def checkpoint(p, state):
    t=p.with_suffix('.tmp'); torch.save(cpu(state), t)
    with t.open('r+b') as f: f.flush(); os.fsync(f.fileno())
    if p.exists(): os.replace(p, p.with_name('previous.pt'))
    os.replace(t, p)

def train(a):
    plan=json.loads((a.runtime/'plan.json').read_text())
    for name,digest in plan.get('source_hashes',{}).items():
        if sha(ROOT/name)!=digest: raise ValueError('Source changed: '+name)
    out=a.runtime/a.variant; out.mkdir(exist_ok=True)
    with lock(out/'run.lock'):
        seed(); model=StructureNet(a.variant).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda'); config=dict(plan=plan, variant=a.variant)
        fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        epoch=0; next_batch=0; total=0.; seen=0; best=float('inf'); elapsed=0.
        if (out/'latest.pt').exists():
            ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
            if ck['fingerprint']!=fingerprint: raise ValueError('Changed run configuration')
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
            epoch=ck['epoch']; next_batch=ck['next_batch']; total=ck['sum_squared']; seen=ck['elements']; best=ck['best_mse']; elapsed=ck['elapsed_seconds']
            random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        train_ds=Cached(a.runtime/'cache/train'); val_ds=Cached(a.runtime/'cache/val')
        val_loader=loader(val_ds,plan['workers']); start=time.monotonic(); last_save=start; last_log=0
        def state(e,b,s,n):
            return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),config=config,fingerprint=fingerprint,
                        epoch=e,next_batch=b,sum_squared=s,elements=n,best_mse=best,elapsed_seconds=elapsed+time.monotonic()-start,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        while epoch<plan['epochs']:
            model.train(); ds_loader=loader(train_ds,plan['workers'],Remaining(len(train_ds),epoch,next_batch,plan['effective_batch']))
            loss_sum=torch.tensor(total,device='cuda',dtype=torch.float64)
            steps=math.ceil(len(train_ds)/plan['effective_batch']); begin=time.monotonic(); begin_batch=next_batch
            for idx,batch in enumerate(ds_loader,start=next_batch):
                optimizer.zero_grad(set_to_none=True); batch=[x.cuda(non_blocking=True) for x in batch]; n=len(batch[0]); denom=batch[3].numel()
                for j in range(0,n,plan['micro']):
                    pan,lms,ms,gt=[x[j:j+plan['micro']] for x in batch]
                    with torch.autocast('cuda',dtype=torch.float16): pred=model(pan,lms,ms)
                    squared=(pred.float()-gt).square().sum()
                    if not torch.isfinite(squared): raise FloatingPointError('Nonfinite train loss')
                    scaler.scale(squared/denom).backward(); loss_sum+=squared.detach().double()
                scaler.step(optimizer); scaler.update(); seen+=denom; next_batch=idx+1
                now=time.monotonic()
                if now-last_log>=5:
                    speed=(next_batch-begin_batch)*plan['effective_batch']/max(now-begin,1e-6)
                    write(out/'status.json',dict(pid=os.getpid(),state='training',variant=a.variant,epoch=epoch+1,epochs=plan['epochs'],batch=next_batch,batches=steps,percent=100*next_batch/steps,patches_per_second=speed,
                        allocated_mib=torch.cuda.memory_allocated()/1024**2,peak_mib=torch.cuda.max_memory_allocated()/1024**2,updated=time.time(),elapsed_seconds=elapsed+now-start))
                    last_log=now
                if now-last_save>=300:
                    checkpoint(out/'latest.pt',state(epoch,next_batch,loss_sum.item(),seen));last_save=time.monotonic()
            del ds_loader
            train_mse=loss_sum.item()/seen; model.eval(); val_sum=torch.zeros((),device='cuda',dtype=torch.float64); val_n=0; angles=torch.zeros((),device='cuda',dtype=torch.float64); pixels=0
            with torch.inference_mode():
                for batch in val_loader:
                    for j in range(0,len(batch[0]),plan['micro']):
                        pan,lms,ms,gt=[x[j:j+plan['micro']].cuda(non_blocking=True) for x in batch]
                        pred=model(pan,lms,ms).float(); val_sum+=(pred-gt).square().sum().double();val_n+=gt.numel()
                        cosine=(pred*gt).sum(1)/(pred.norm(dim=1)*gt.norm(dim=1)).clamp_min(1e-12)
                        mask=(gt.norm(dim=1)>1e-8)&(pred.norm(dim=1)>1e-8)
                        angles+=torch.acos(cosine.clamp(-1,1))[mask].sum().double();pixels+=mask.sum().item()
            mse=val_sum.item()/val_n; sam=math.degrees(angles.item()/max(pixels,1))
            if not math.isfinite(mse): raise FloatingPointError('Nonfinite validation')
            improved=mse<best; best=min(best,mse); epoch+=1;next_batch=0;total=0.;seen=0
            row=dict(epoch=epoch,train_mse=train_mse,val_mse=mse,val_sam=sam,val_global_psnr=-10*math.log10(max(mse,1e-15)),best_mse=best,seconds=elapsed+time.monotonic()-start)
            history_path=out/'history.json'; hist=json.loads(history_path.read_text()) if history_path.exists() else []
            hist=[r for r in hist if r['epoch']<epoch]+[row];write(history_path,hist)
            st=state(epoch,0,0.,0)
            if improved:
                t=out/'best.tmp';torch.save(cpu(dict(st,best_epoch=epoch,val_sam=sam)),t);os.replace(t,out/'best.pt')
            checkpoint(out/'latest.pt',st)
            print(json.dumps(row),flush=True);last_save=time.monotonic()
        write(out/'complete.json',dict(variant=a.variant,epochs=epoch,best_mse=best,best_sha256=sha(out/'best.pt'),latest_sha256=sha(out/'latest.pt'),fingerprint=fingerprint))
        write(out/'status.json',dict(state='complete',epoch=epoch,updated=time.time(),pid=os.getpid()))

def cache(a):
    import h5py
    result={}
    for split,name in [('train','train_qb.h5'),('val','valid_qb.h5')]:
        matches=[p for p in a.data.rglob(name) if p.name==name]
        if len(matches)!=1: raise ValueError(f'{name}: {matches}')
        path=matches[0]; digest=sha(path); d=a.runtime/'cache'/split;d.mkdir(parents=True,exist_ok=True)
        with h5py.File(path) as f:
            if len(f['pan']) != (17139 if split=='train' else 1905): raise ValueError('Unexpected QB split size')
            result[split]=dict(path=str(path),sha256=digest,shapes={k:list(f[k].shape) for k in KEYS})
            for k in KEYS:
                tmp=d/(k+'.tmp.npy');arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=f[k].shape)
                for i in range(0,len(f[k]),128):
                    values=np.asarray(f[k][i:i+128],dtype=np.float32)/2047.
                    if not np.isfinite(values).all():raise ValueError('Nonfinite source')
                    arr[i:i+128]=values
                arr.flush();del arr;os.replace(tmp,d/(k+'.npy'))
                result[split].setdefault('cache_hashes',{})[k]=sha(d/(k+'.npy'))
    write(a.runtime/'data_manifest.json',result)

def probe(a):
    seed(); model=StructureNet(a.variant).cuda(); opt=torch.optim.Adam(model.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda')
    ds=Cached(a.runtime/'cache/train'); batch=[x.unsqueeze(0).repeat(a.micro,1,1,1).cuda() for x in ds[0]]
    # Separate initializations from trained state. All candidates receive gradient/shape checks.
    real_loader=DataLoader(ds,batch_size=a.micro,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=1) if a.real_data else None
    real_iter=iter(real_loader) if real_loader else None
    t=time.monotonic()
    for step in range(24):
        if real_iter is not None: batch=[x.cuda(non_blocking=True) for x in next(real_iter)]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): pred=model(*batch[:3])
        loss=(pred.float()-batch[3]).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite probe')
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        if step==7:torch.cuda.synchronize();t=time.monotonic()
    torch.cuda.synchronize()
    write(a.runtime/f'probe_{a.variant}_{a.micro}.json',dict(variant=a.variant,micro=a.micro,patches_per_second=16*a.micro/(time.monotonic()-t),peak_mib=torch.cuda.max_memory_reserved()/1024**2,loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters())))

def controller(a):
    with lock(a.runtime/'controller.lock'):
        write(a.runtime/'controller.json',dict(pid=os.getpid(),started=time.time()))
        active={};failed=[v for v in VARIANTS if (a.runtime/v/'failed.json').exists()]
        while True:
            for v,(p,f) in list(active.items()):
                if p.poll() is None: continue
                f.close();del active[v]
                if p.returncode or not (a.runtime/v/'complete.json').exists():
                    failed.append(v);write(a.runtime/v/'failed.json',dict(exitcode=p.returncode,time=time.time()))
                    if v=='S0':raise RuntimeError('Baseline failed; queue stopped, active candidate remains untouched')
            completed=[v for v in VARIANTS if (a.runtime/v/'complete.json').exists()]
            pending=[v for v in VARIANTS if v not in active and v not in completed and v not in failed]
            for v in pending:
                if len(active)>=2:break
                if v=='M1' and 'S3' not in completed:
                    if 'S3' in failed: failed.append(v);write(a.runtime/v/'failed.json',dict(reason='S3 dependency failed'))
                    continue
                d=a.runtime/v;d.mkdir(exist_ok=True)
                # A stale controller must not launch duplicates over a surviving child.
                s=d/'status.json'
                if s.exists():
                    old=json.loads(s.read_text())
                    if old.get('state')=='training' and alive(old.get('pid',0)):raise RuntimeError(f'Existing child {v}; wait or attach before restart')
                f=(d/'train.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,__file__,'train','--runtime',str(a.runtime),'--variant',v],stdout=f,stderr=subprocess.STDOUT)
                active[v]=(p,f)
            write(a.runtime/'queue.json',dict(active={v:p.pid for v,(p,f) in active.items()},completed=completed,failed=failed,updated=time.time()))
            if not active:
                if failed:raise RuntimeError('Incomplete suite; final selection withheld')
                evaluate(a);break
            time.sleep(5)

def alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
        if h:ctypes.windll.kernel32.CloseHandle(h);return bool(h)
        return False
    try:os.kill(pid,0);return True
    except OSError:return False

def evaluate(a):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from common.metrics import rr_metrics, fr_metrics
    from common.preview import panels
    seed();rank=[]
    for v in VARIANTS:
        ck=torch.load(a.runtime/v/'best.pt',map_location='cpu',weights_only=False)
        rank.append(dict(variant=v,best_mse=ck['best_mse'],val_sam=ck['val_sam'],epoch=ck['best_epoch']))
    rank.sort(key=lambda x:(x['best_mse'],x['val_sam'],VARIANTS.index(x['variant'])))
    write(a.runtime/'ranking.json',dict(selection='validation MSE then validation SAM; exact tie uses fixed queue order',rows=rank))
    plan=json.loads((a.runtime/'plan.json').read_text());data=Path(plan['data']);out=a.runtime/'evaluation';out.mkdir(exist_ok=True)
    selection=dict(seed=42,indices=[1,8,11,13,19],order='ascending',tile='full scene',frozen_before_test=True)
    write(out/'selection.json',selection);report={}
    candidate=next(row['variant'] for row in rank if row['variant']!='S0')
    for v in ['S0',candidate]:
        ck=torch.load(a.runtime/v/'best.pt',map_location='cpu',weights_only=False);m=StructureNet(v).cuda().eval();m.load_state_dict(ck['model'])
        record=dict(best_epoch=ck['best_epoch'],best_sha256=sha(a.runtime/v/'best.pt'),metrics={})
        for protocol,name in [('RR','test_qb_multiExm1.h5'),('FR','test_qb_OrigScale_multiExm1.h5')]:
            paths=list(data.rglob(name))
            if len(paths)!=1:raise ValueError(paths)
            rows=[]
            with h5py.File(paths[0]) as f:
                if len(f['pan'])!=20:raise ValueError('Expected20 scenes')
                for i in range(20):
                    arrays={k:np.asarray(f[k][i],dtype=np.float32) for k in f.keys()}
                    inp=[torch.from_numpy(arrays[k]/2047.).unsqueeze(0).cuda() for k in KEYS[:3]]
                    with torch.inference_mode():pred=m(*inp)[0].cpu().numpy().astype(np.float64)*2047.
                    metrics=rr_metrics(arrays['gt'].astype(np.float64),pred) if protocol=='RR' else fr_metrics(pred,arrays['lms'],arrays['ms'],arrays['pan'])
                    if protocol=='RR':
                        diff=(pred-arrays['gt'])/2047.
                        metrics['MSE_peak1']=float(np.mean(diff**2))
                        gx=np.diff(arrays['gt'].mean(0),axis=1,prepend=arrays['gt'].mean(0)[:,:1]);gy=np.diff(arrays['gt'].mean(0),axis=0,prepend=arrays['gt'].mean(0)[:1,:]);edge=np.hypot(gx,gy);mask=edge>=np.percentile(edge,75)
                        metrics['edge_MSE_peak1']=float(np.mean(diff[:,mask]**2))
                    else:
                        reduced=torch.nn.functional.interpolate(torch.from_numpy(pred).unsqueeze(0),size=arrays['ms'].shape[1:],mode='bilinear',align_corners=True)[0].numpy()
                        metrics['LR_MSE_bilinear_peak1']=float(np.mean(((reduced-arrays['ms'])/2047.)**2))
                    if not all(np.isfinite(x) for x in metrics.values()):raise ValueError('Nonfinite test')
                    rows.append(dict(index=i,metrics=metrics))
                    if protocol=='RR' and i in selection['indices']:panels(arrays,pred,out/f'{v}_scene_{i:02}.png',v,i)
            record['metrics'][protocol]=dict(data_sha256=sha(paths[0]),rows=rows,mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']})
        report[v]=record;del m;torch.cuda.empty_cache();write(out/'metrics.json',dict(runs=report,limitations=['FR QNR and Q2n/SCC not MATLAB parity verified','Single QB seed50epoch screening'],selection=selection))
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    for v in VARIANTS:
        h=json.loads((a.runtime/v/'history.json').read_text());ep=[r['epoch'] for r in h]
        for ax,key in zip(axes,['train_mse','val_mse','val_global_psnr','val_sam']):ax.plot(ep,[r[key] for r in h],label=v);ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[1].set_yscale('log');axes[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,(protocol,key) in zip(axes,[('RR','PSNR'),('RR','SAM'),('RR','MSE_peak1'),('FR','QNR')]):
        delta=report[candidate]['metrics'][protocol]['mean'][key]-report['S0']['metrics'][protocol]['mean'][key]
        ax.bar([candidate],[delta]);ax.axhline(0,color='black',linewidth=.8);ax.set_title(protocol+' '+key+' delta vs S0')
    fig.tight_layout();fig.savefig(out/'test_comparison.png',dpi=160);plt.close(fig)
    write(a.runtime/'suite_complete.json',dict(time=time.time(),runs=list(VARIANTS),winner=rank[0]['variant'],epochs=50,evaluation='evaluation/metrics.json'))

def start(a):
    if os.name!='nt': raise RuntimeError('Windows launcher only')
    if (a.runtime/'controller.json').exists() and alive(json.loads((a.runtime/'controller.json').read_text()).get('pid')): raise RuntimeError('Controller already active')
    import base64
    command=subprocess.list2cmdline([sys.executable,__file__,'controller','--runtime',str(a.runtime)])
    line='cmd.exe /d /c "'+command+' >> "'+str(a.runtime/'controller.log')+'" 2>&1"'
    ps="$ProgressPreference='SilentlyContinue'; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+line.replace("'","''")+"'} | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],capture_output=True,text=True,check=True)
    result=json.loads(r.stdout)
    if result['ReturnValue']!=0:raise RuntimeError(result)
    print(json.dumps(result),flush=True)

def status(a):
    q=a.runtime/'queue.json'
    print(q.read_text() if q.exists() else 'Queue has not started')
    for v in VARIANTS:
        p=a.runtime/v/'status.json'
        if p.exists():print(v,p.read_text())

def logs(a):
    while True:
        items=[]
        for v in VARIANTS:
            p=a.runtime/v/'status.json'
            if p.exists():
                s=json.loads(p.read_text())
                if s.get('state')=='training': items.append(f"{v} epoch {s['epoch']}/50 {s['percent']:.1f}% {s['patches_per_second']:.0f} patch/s")
        print('\r'+' | '.join(items).ljust(150),end='',flush=True)
        time.sleep(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','probe','train','controller','evaluate','start','status','logs']);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--variant',choices=VARIANTS,default='S0');p.add_argument('--micro',type=int,default=8);p.add_argument('--real-data',action='store_true');a=p.parse_args();a.runtime.mkdir(parents=True,exist_ok=True)
    globals()[a.mode](a)
if __name__=='__main__':main()
