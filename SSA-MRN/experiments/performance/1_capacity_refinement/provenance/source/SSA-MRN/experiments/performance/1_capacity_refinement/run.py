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
StructureNet = __import__('importlib').import_module('experiments.performance.1_capacity_refinement.model').StructureNet; VARIANTS=('R0','W2','W3','F01','F11')
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
        self.folder=Path(folder); self.arrays=None
        self.n=len(np.load(self.folder/'pan.npy',mmap_mode='r'))
    def __len__(self): return self.n
    def __getitem__(self, indices):
        if self.arrays is None: self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in KEYS}
        # Batch indices: one contiguous FP32 copy per key instead of one copy per patch.
        return tuple(torch.from_numpy(np.array(self.arrays[k][indices],copy=True) if isinstance(indices,int) else np.asarray(self.arrays[k][indices])) for k in KEYS)
    def __getstate__(self): return dict(folder=self.folder,n=self.n,arrays=None)

class Remaining:
    def __init__(self, n, epoch, start, batch=32): self.n=n; self.epoch=epoch; self.start=start; self.batch=batch
    def __iter__(self):
        order=torch.randperm(self.n, generator=torch.Generator().manual_seed(42+self.epoch)).tolist()
        for i in range(self.start*self.batch, self.n, self.batch): yield order[i:i+self.batch]
    def __len__(self): return math.ceil(self.n/self.batch)-self.start

def loader(ds, workers, sampler=None):
    opts=dict(num_workers=workers, pin_memory=True)
    if workers: opts.update(persistent_workers=True, prefetch_factor=1)
    if sampler is None: sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(ds),64,False)
    return DataLoader(ds,batch_size=None,sampler=sampler,**opts)

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
        seed(); model=StructureNet(a.variant,plan['channels']); model.configure(plan['calibration']); model=model.cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda'); config=dict(plan=plan, variant=a.variant)
        fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        initial_state={k:v for k,v in model.state_dict().items()}
        digest=hashlib.sha256()
        for name,tensor in sorted(initial_state.items()): digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        # Common initialization is independently verified in preflight.
        write(out/'initialization.json',dict(core_sha256=digest.hexdigest(),parameters=sum(p.numel() for p in model.parameters())))
        updates=[0]; optimizer.register_step_post_hook(lambda *args:updates.__setitem__(0,updates[0]+1))
        skipped_steps=0
        epoch=0; next_batch=0; total=0.; seen=0; best=float('inf'); best_sam=float('inf'); elapsed=0.; best_epoch=0
        resume_path=out/'latest.pt' if (out/'latest.pt').exists() else out/'previous.pt'
        if resume_path.exists():
            try: ck=torch.load(resume_path,map_location='cpu',weights_only=False)
            except Exception:
                fallback=out/'previous.pt'
                if fallback==resume_path or not fallback.exists():raise
                ck=torch.load(fallback,map_location='cpu',weights_only=False)
            if ck['fingerprint']!=fingerprint: raise ValueError('Changed run configuration')
            write(out/'history.json',ck.get('history',[])); model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
            skipped_steps=ck.get('skipped_steps',0)
            epoch=ck['epoch']; next_batch=ck['next_batch']; total=ck['sum_squared']; seen=ck['elements']; best=ck['best_mse']; elapsed=ck['elapsed_seconds']; best_sam=ck.get('best_sam',float('inf'))
            best_epoch=ck.get('best_epoch',min(ck['history'],key=lambda r:(r['val_mse'],r['val_sam'],r['epoch']))['epoch'] if ck.get('history') else 0)
            random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        cache_root=Path(plan.get('cache_root',a.runtime/'cache')); train_ds=Cached(cache_root/'train'); val_ds=Cached(cache_root/'val')
        val_loader=loader(val_ds,plan['workers']); sampler=Remaining(len(train_ds),epoch,next_batch,plan['effective_batch']); ds_loader=loader(train_ds,plan['workers'],sampler); start=time.monotonic(); last_save=start; last_log=0
        def state(e,b,s,n):
            return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),config=config,fingerprint=fingerprint,
                        epoch=e,best_epoch=best_epoch,next_batch=b,sum_squared=s,elements=n,best_mse=best,best_sam=best_sam,elapsed_seconds=elapsed+time.monotonic()-start,
                        skipped_steps=skipped_steps,history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else [],python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        while epoch<a.target:
            for group in optimizer.param_groups:group['lr']=1e-6+(1e-4-1e-6)*0.5*(1+math.cos(math.pi*epoch/149))
            model.train(); sampler.epoch=epoch; sampler.start=next_batch
            loss_sum=torch.tensor(total,device='cuda',dtype=torch.float64)
            steps=math.ceil(len(train_ds)/plan['effective_batch']); begin=time.monotonic(); begin_batch=next_batch
            for idx,batch in enumerate(ds_loader,start=next_batch):
                optimizer.zero_grad(set_to_none=True); batch=[x.cuda(non_blocking=True) for x in batch]; n=len(batch[0]); denom=batch[3].numel()
                for j in range(0,n,plan['micro']):
                    pan,lms,ms,gt=[x[j:j+plan['micro']] for x in batch]
                    with torch.autocast('cuda',dtype=torch.float16): pred=model(pan,lms,ms)
                    squared=(pred.float()-gt).square().sum()
                    # GradScaler checks gradients; accumulated loss is checked at status/save boundaries.
                    scaler.scale(squared/denom).backward(); loss_sum+=squared.detach().double()
                before=updates[0];scaler.step(optimizer);scaler.update();skipped_steps+=int(updates[0]==before);seen+=denom; next_batch=idx+1
                now=time.monotonic()
                if now-last_log>=5:
                    if not torch.isfinite(loss_sum): raise FloatingPointError('Nonfinite train loss')
                    speed=(next_batch-begin_batch)*plan['effective_batch']/max(now-begin,1e-6)
                    write(out/'status.json',dict(pid=os.getpid(),state='training',variant=a.variant,epoch=epoch+1,epochs=plan['epochs'],batch=next_batch,batches=steps,percent=100*next_batch/steps,patches_per_second=speed,
                        allocated_mib=torch.cuda.memory_allocated()/1024**2,peak_mib=torch.cuda.max_memory_allocated()/1024**2,updated=time.time(),elapsed_seconds=elapsed+now-start))
                    last_log=now
                if now-last_save>=300:
                    checkpoint(out/'latest.pt',state(epoch,next_batch,loss_sum.item(),seen));last_save=time.monotonic()
            train_mse=loss_sum.item()/seen
            if not math.isfinite(train_mse): raise FloatingPointError('Nonfinite train epoch')
            write(out/'status.json',dict(pid=os.getpid(),state='validation',variant=a.variant,epoch=epoch+1,epochs=plan['epochs'],updated=time.time()))
            model.eval(); val_sum=torch.zeros((),device='cuda',dtype=torch.float64); val_n=0; angles=torch.zeros((),device='cuda',dtype=torch.float64); pixels=0; band_psnr_sum=torch.zeros((),device='cuda',dtype=torch.float64); samples=0
            with torch.inference_mode():
                for batch in val_loader:
                    for j in range(0,len(batch[0]),plan['micro']):
                        pan,lms,ms,gt=[x[j:j+plan['micro']].cuda(non_blocking=True) for x in batch]
                        pred=model(pan,lms,ms).float(); band_mse=(pred-gt).square().mean((2,3)); band_psnr_sum+=(-10*torch.log10(band_mse.clamp_min(1e-15))).mean(1).sum().double(); samples+=len(gt); val_sum+=(pred-gt).square().sum().double();val_n+=gt.numel()
                        cosine=(pred*gt).sum(1)/(pred.norm(dim=1)*gt.norm(dim=1)).clamp_min(1e-12)
                        mask=(gt.norm(dim=1)>1e-8)&(pred.norm(dim=1)>1e-8)
                        angles+=torch.acos(cosine.clamp(-1,1))[mask].sum().double();pixels+=mask.sum().item()
            mse=val_sum.item()/val_n; sam=math.degrees(angles.item()/max(pixels,1))
            if not math.isfinite(mse): raise FloatingPointError('Nonfinite validation')
            improved=mse<best
            if improved: best=mse; best_sam=sam; best_epoch=epoch+1
            epoch+=1;next_batch=0;total=0.;seen=0
            row=dict(epoch=epoch,lr=optimizer.param_groups[0]['lr'],skipped_steps=skipped_steps,train_mse=train_mse,val_mse=mse,val_sam=sam,val_global_psnr=-10*math.log10(max(mse,1e-15)),val_psnr_band_mean=band_psnr_sum.item()/samples,best_mse=best,seconds=elapsed+time.monotonic()-start,gates=model.gate_values())
            history_path=out/'history.json'; hist=json.loads(history_path.read_text()) if history_path.exists() else []
            hist=[r for r in hist if r['epoch']<epoch]+[row];write(history_path,hist)
            st=state(epoch,0,0.,0)
            if improved:
                t=out/'best.tmp';torch.save(cpu(dict(st,best_epoch=epoch,val_sam=sam)),t);os.replace(t,out/'best.pt')
            checkpoint(out/'latest.pt',st)
            if epoch in (50,100,150):
                import shutil
                shutil.copy2(out/'best.pt',out/f'best_at_{epoch}.pt')
            print(json.dumps(row),flush=True);last_save=time.monotonic()
        import shutil
        shutil.copy2(out/'best.pt',out/f'best_at_{a.target}.pt')
        write(out/f'complete_{a.target}.json',dict(variant=a.variant,epochs=epoch,best_epoch=best_epoch,best_mse=best,best_sha256=sha(out/f'best_at_{a.target}.pt'),latest_sha256=sha(out/'latest.pt'),fingerprint=fingerprint))
        write(out/'status.json',dict(state='complete',epoch=epoch,updated=time.time(),pid=os.getpid()))


SENSORS={'QB':4}

def probe(a):
    plan=json.loads((a.runtime/'QB/plan.json').read_text());seed();m=StructureNet(a.variant,4);m.configure(plan['calibration']);m=m.cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda')
    batch=[x.unsqueeze(0).repeat(plan['micro'],1,1,1).cuda() for x in Cached(Path(plan['cache_root'])/'train')[0]]
    start=time.monotonic()
    for i in range(15):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):pred=m(*batch[:3])
        loss=(pred.float()-batch[3]).square().mean();assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
    m.eval()
    with torch.inference_mode():assert torch.isfinite(m(*batch[:3])).all()
    torch.cuda.synchronize();write(a.runtime/('probe_'+a.variant+'.json'),dict(micro=plan['micro'],peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,seconds=time.monotonic()-start,loss=float(loss)))

def evaluate(a):
    import h5py
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from common.metrics import rr_metrics,fr_metrics
    plan=json.loads((a.runtime/'plan.json').read_text());manifest=json.loads((a.runtime/'data_manifest.json').read_text());out=a.runtime/'evaluation'/str(a.target);out.mkdir(parents=True,exist_ok=True);report={}
    for v in plan.get('variants',VARIANTS):
        path=a.runtime/v/f'best_at_{a.target}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);m=StructureNet(v,plan['channels']);m.configure(plan['calibration']);m=m.cuda().eval();m.load_state_dict(ck['model'],strict=True)
        record={'best_epoch':ck['best_epoch'],'best_sha256':sha(path),'metrics':{}}
        for protocol in ('RR','FR'):
            rows=[]
            with h5py.File(manifest[protocol]['path']) as f:
                indices=[i for i in [1,8,11,13,19] if i<len(f['pan'])]
                for i in range(len(f['pan'])):
                    arrays={k:np.asarray(f[k][i],dtype=np.float32) for k in f}
                    inp=[torch.from_numpy(arrays[k]/plan['peak']).unsqueeze(0).cuda() for k in KEYS[:3]]
                    with torch.inference_mode():pred=m(*inp)[0].cpu().numpy().astype(np.float64)*plan['peak']
                    metrics=rr_metrics(arrays['gt'].astype(np.float64)*2047./plan['peak'],pred*2047./plan['peak']) if protocol=='RR' else fr_metrics(pred,arrays['lms'],arrays['ms'],arrays['pan'])
                    if not all(np.isfinite(x) for x in metrics.values()):raise ValueError('Nonfinite test metrics')
                    rows.append({'index':i,'metrics':metrics})
                    if protocol=='RR' and i in indices:
                        bands=[4,2,1] if plan['sensor']=='WV3' else [2,1,0]
                        ref=arrays['gt'][bands].transpose(1,2,0);lo,hi=np.percentile(ref,[1,99],axis=(0,1))
                        def rgb(x):return Image.fromarray((np.clip((x[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
                        pn=arrays['pan'][0];a0,a1=np.percentile(pn,[1,99]);pi=Image.fromarray((np.clip((pn-a0)/max(a1-a0,1e-8),0,1)*255).astype('uint8')).convert('RGB');h,w=pn.shape
                        canvas=Image.new('RGB',(4*w,h+38),'white');draw=ImageDraw.Draw(canvas)
                        for j,(label,img) in enumerate([('LR MS',rgb(arrays['ms'])),('PAN',pi),(v,rgb(pred)),('GT',rgb(arrays['gt']))]):canvas.paste(img.resize((w,h),Image.Resampling.NEAREST),(j*w,38));draw.text((j*w+8,8),label,fill='black')
                        canvas.save(out/f'{v}_scene_{i:02}.png')
            record['metrics'][protocol]={'rows':rows,'mean':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
        report[v]=record;del m;torch.cuda.empty_cache();write(out/'metrics.json',{'runs':report,'selection':'validation best up to milestone, never test','limitations':['QNR/Q2n/SCC MATLAB parity unverified'],'scene_indices':indices,'rgb_assumed_bands':bands})
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for v in plan.get('variants',VARIANTS):
        hist=[r for r in json.loads((a.runtime/v/'history.json').read_text()) if r['epoch']<=a.target]
        for ax,k in zip(axes,('train_mse','val_mse','val_psnr_band_mean','val_sam')):ax.plot([r['epoch'] for r in hist],[r[k] for r in hist],label=v);ax.set_title(k);ax.grid(alpha=.2)
    axes[-1].legend();fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    vs=plan['variants']; baseline=report['R0']
    for ax,(proto,k) in zip(axes,[('RR','PSNR'),('RR','SAM'),('FR','QNR')]):
        values=[report[v]['metrics'][proto]['mean'][k]-baseline['metrics'][proto]['mean'][k] for v in vs]
        ax.bar(vs,values);ax.set_title(f'{proto} {k} delta vs R0');ax.axhline(0,color='black',lw=.8)
    fig.tight_layout();fig.savefig(out/'comparison.png');plt.close(fig)
    write(out/'complete.json',{'target':a.target,'sensor':plan['sensor'],'evaluated':plan['variants'],'time':time.time()})

def controller(a):
    with lock(a.runtime/'controller.lock'):
        write(a.runtime/'controller.json',{'pid':os.getpid(),'started':time.time()})
        plan=json.loads((a.runtime/'QB/plan.json').read_text())
        for target in (150,):
            for sensor in SENSORS:
                run=a.runtime/sensor
                if (run/'evaluation'/str(target)/'complete.json').exists():continue
                active={}
                while True:
                    for v,(p,f) in list(active.items()):
                        if p.poll() is None:continue
                        f.close();del active[v]
                        if p.returncode or not (run/v/f'complete_{target}.json').exists():
                            write(a.runtime/'queue.json',{'state':'failed','sensor':sensor,'target':target,'variant':v,'exitcode':p.returncode,'active':{k:q.pid for k,(q,f) in active.items()}})
                            # Wait for the other job to finish before exiting; no orphan training.
                            for q,f in active.values():q.wait();f.close()
                            raise RuntimeError(f'{sensor} {v} failed')
                    complete=[v for v in plan['variants'] if (run/v/f'complete_{target}.json').exists()]
                    for v in plan.get('variants',VARIANTS):
                        if len(active)>=2:break
                        if v in complete or v in active:continue
                        d=run/v;d.mkdir(exist_ok=True);s=d/'status.json'
                        if s.exists() and alive(json.loads(s.read_text()).get('pid',0)):raise RuntimeError('Surviving child; do not launch duplicate')
                        f=(d/'train.log').open('a',encoding='utf-8');q=subprocess.Popen([sys.executable,__file__,'train','--runtime',str(run),'--variant',v,'--target',str(target)],stdout=f,stderr=subprocess.STDOUT);active[v]=(q,f)
                    write(a.runtime/'queue.json',{'state':'training','sensor':sensor,'target':target,'active':{v:q.pid for v,(q,f) in active.items()},'completed':complete,'updated':time.time()})
                    if not active:break
                    time.sleep(5)
                write(a.runtime/'queue.json',{'state':'evaluation','sensor':sensor,'target':target,'active':{},'updated':time.time()})
                for milestone in (50,100,150):
                    subprocess.run([sys.executable,__file__,'evaluate','--runtime',str(run),'--target',str(milestone)],check=True)
            write(a.runtime/f'milestone_{target}.json',{'state':'complete','target':target,'sensors':list(SENSORS),'time':time.time()})
        write(a.runtime/'queue.json',{'state':'complete','active':{},'time':time.time()})

def alive(pid):
    if not pid:return False
    try:
        import ctypes
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
        if not h:return False
        ctypes.windll.kernel32.CloseHandle(h);return True
    except Exception:return False

def start(a):
    if (a.runtime/'controller.json').exists() and alive(json.loads((a.runtime/'controller.json').read_text()).get('pid',0)):raise RuntimeError('Controller already running')
    import base64
    command=subprocess.list2cmdline([sys.executable,__file__,'controller','--runtime',str(a.runtime)])
    line='cmd.exe /d /c "'+command+' >> "'+str(a.runtime/'controller.log')+'" 2>&1"'
    ps="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+line.replace("'","''")+"'} | ConvertTo-Json -Compress"
    print(subprocess.run(['powershell','-NoProfile','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],capture_output=True,text=True,check=True).stdout)

def logs(a):
    while True:
        q=json.loads((a.runtime/'queue.json').read_text());parts=[q['state'],q.get('sensor',''),str(q.get('target',''))]
        for v in q.get('active',{}):
            p=a.runtime/q['sensor']/v/'status.json'
            if p.exists():
                s=json.loads(p.read_text());parts.append(f"{v}: {s.get('epoch')}/150 {s.get('percent',0):.1f}% {s.get('patches_per_second',0):.0f} patch/s")
        print('\r'+' | '.join(parts).ljust(180),end='',flush=True)
        if not a.follow or q['state'] in ('complete','failed'):print();break
        time.sleep(2)


def prepare(a):
    import h5py
    old=json.loads(Path('C:/CtrS-ssai5/runtime/qb_s42_100/plan.json').read_text())
    cache=Path(old['cache_root']);run=a.runtime/'QB';run.mkdir(exist_ok=True)
    for split in ('train','val'):
        for key,digest in old['data'][split]['cache'].items():
            if sha(cache/split/(key+'.npy'))!=digest:raise ValueError('Cache hash mismatch')
    data=Path(old['data_root']);manifest={}
    for proto,pattern in [('RR','test_qb_multiExm1.h5'),('FR','test_qb_OrigScale_multiExm1.h5')]:
        paths=[p for p in data.rglob(pattern) if not p.name.startswith('._')];assert len(paths)==1,paths
        with h5py.File(paths[0]) as f:assert len(f['pan'])==20
        manifest[proto]=dict(path=str(paths[0]),sha256=sha(paths[0]))
    plan=dict(seed=42,sensor='QB',channels=4,peak=2047.,epochs=150,milestones=[50,100,150],effective_batch=32,micro=32,workers=1,max_parallel=2,variants=list(VARIANTS),cache_root=str(cache),calibration=json.loads(Path(__file__).with_name('calibration.json').read_text()),selection='minimum validation MSE; earliest exact tie',precision='AMP FP16 train; FP32 validation/test',schedule='cosine epoch0 LR1e-4 to epoch149 LR1e-6',source_hashes={})
    paths=list((ROOT/'common').glob('*.py'))+[ROOT/'references/upstream/network.py',Path(__file__),Path(__file__).with_name('model.py'),ROOT/'research/performance/17_structural_ablation/model.py',ROOT/'research/performance/28_residual_refinement/model.py']
    plan['source_hashes']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    write(run/'data_manifest.json',manifest);write(run/'plan.json',plan);write(a.runtime/'suite_plan.json',plan)
    print('PREPARED verified cache and RR20/FR20',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','probe','train','evaluate','controller','start','logs']);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--variant',choices=VARIANTS,default='R0');p.add_argument('--target',type=int,choices=[50,100,150],default=150);p.add_argument('--micro',type=int,default=32);p.add_argument('--follow',action='store_true');a=p.parse_args();a.runtime.mkdir(parents=True,exist_ok=True);globals()[a.mode](a)
if __name__=='__main__':main()
