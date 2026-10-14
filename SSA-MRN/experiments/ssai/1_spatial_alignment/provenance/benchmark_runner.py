"""Fixed-condition SSAI suite: shared mmap, benchmarked concurrency, detached WMI."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,base64,contextlib,hashlib,importlib,json,math,random,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common.model import RestoredPansharpeningNet
ARMS=('B0','A1','G1')
KEYS=('pan','lms','ms','gt')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)


def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(cpu(v)for v in x)
    return x


def save(p,data):
    tmp=p.with_suffix('.tmp');torch.save(cpu(data),tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,p)


@contextlib.contextmanager
def lock(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
        if not h:return False
        ctypes.windll.kernel32.CloseHandle(h);return True
    try:os.kill(pid,0);return True
    except OSError:return False


def configure(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':torch.cuda.manual_seed_all(seed)


def model(arm,channels=4):
    if arm=='B0':return RestoredPansharpeningNet(channels,4)
    name,cls={'A1':('1_spatial_alignment','SpatialAlignmentNet'),'R1':('2_reference_area_normalization','ReferenceAreaNet'),'G1':('3_calibrated_sigmoid_gating','CalibratedSigmoidNet')}[arm]
    return getattr(importlib.import_module(f'research.ssai.{name}.model'),cls)(channels,4)


class Cached(Dataset):
    def __init__(self,path):
        self.arrays={k:np.load(Path(path)/(k+'.npy'),mmap_mode='r')for k in KEYS}
    def __len__(self):return len(self.arrays['pan'])
    def __getitem__(self,i):return tuple(torch.from_numpy(np.array(self.arrays[k][i],copy=True))for k in KEYS)


def plan(a):
    p=json.loads((a.runtime/'plan.json').read_text())
    for name,h in p['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Source changed: '+name)
    return p


def prepare(a):
    import h5py
    if (a.runtime/'plan.json').exists():raise FileExistsError('Use a fresh runtime')
    if shutil.disk_usage(a.runtime).free<20*1024**3:raise RuntimeError('At least 20 GiB free required')
    metadata={}
    for split,filename in [('train','train_qb.h5'),('val','valid_qb.h5')]:
        matches=list(a.data.rglob(filename))
        if len(matches)!=1:raise ValueError(matches)
        path=matches[0];dst=a.runtime/'cache'/split;dst.mkdir(parents=True)
        with h5py.File(path,'r') as f:
            n=17139 if split=='train' else 1905
            expected={'pan':(n,1,64,64),'lms':(n,4,64,64),'ms':(n,4,16,16),'gt':(n,4,64,64)}
            if any(tuple(f[k].shape)!=shape for k,shape in expected.items()):raise ValueError('Unexpected original QB split geometry')
            metadata[split]={'path':str(path),'sha256':sha(path),'shapes':{k:list(f[k].shape)for k in KEYS},'cache':{}}
            for k in KEYS:
                tmp=dst/(k+'.tmp.npy');arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=f[k].shape)
                for i in range(0,n,128):
                    v=np.asarray(f[k][i:i+128],dtype=np.float32)/2047.
                    if not np.isfinite(v).all():raise ValueError('Nonfinite data')
                    arr[i:i+128]=v
                arr.flush();del arr;os.replace(tmp,dst/(k+'.npy'))
                metadata[split]['cache'][k]=sha(dst/(k+'.npy'))
    files=[ROOT/'common/ssai_suite.py',ROOT/'common/model.py',ROOT/'common/data.py',ROOT/'common/metrics.py',ROOT/'common/preview.py',ROOT/'references/upstream/network.py']
    files+=list((ROOT/'research/ssai').glob('*/model.py'))+list((ROOT/'research/ssai').glob('*/config.json'))
    hashes={str(p.relative_to(ROOT)):sha(p)for p in files}
    for p in files:
        dst=a.runtime/'provenance/source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)if torch.cuda.is_available()else None}
    p={'seed':42,'sensor':'QB','k':4,'epochs':100,'effective_batch':32,'micro_batch':4,'lr':1e-4,'precision':'FP32','amp':False,'deterministic':True,'tf32':False,'data_root':str(a.data),'data':metadata,'source_hashes':hashes,'environment':environment,'arms':list(ARMS),'reference_area_candidate':'same B0 best, no separate training','selection':'min validation MSE, earliest exact tie; no test selection','resume':'exact epoch-boundary state; interruption replays only incomplete epoch','created_utc':time.time()}
    write(a.runtime/'plan.json',p);write(a.runtime/'provenance/plan.json',p)
    print(json.dumps({'prepared':True,'train':17139,'val':1905,'environment':environment}),flush=True)


def run_epoch(m,opt,loader,device,limit=None):
    m.train();total=0.;samples=0;started=time.monotonic()
    for idx,b in enumerate(loader):
        if limit is not None and idx>=limit:break
        b=[x.to(device,non_blocking=device=='cuda')for x in b];n=len(b[0]);opt.zero_grad(set_to_none=True)
        step_sum=torch.zeros((),device=device,dtype=torch.float64)
        for off in range(0,n,4):
            pan,lms,ms,gt=[x[off:off+4]for x in b]
            mse=torch.nn.functional.mse_loss(m(pan,lms,ms),gt)
            (mse*(len(gt)/n)).backward();step_sum+=mse.detach().double()*len(gt)
        value=step_sum.item()
        if not math.isfinite(value):raise FloatingPointError('Nonfinite train loss')
        opt.step();total+=value;samples+=n
        yield idx,total,samples,started


def validate(m,loader,device):
    m.eval();s=0.;n=0;angles=0.;pixels=0
    with torch.inference_mode():
        for b in loader:
            pan,lms,ms,gt=[x.to(device,non_blocking=device=='cuda')for x in b];pred=m(pan,lms,ms)
            s+=torch.nn.functional.mse_loss(pred,gt).item()*len(gt);n+=len(gt)
            norms=pred.norm(dim=1)*gt.norm(dim=1);mask=norms>1e-12
            cosine=(pred*gt).sum(1)/norms.clamp_min(1e-12)
            angles+=torch.acos(cosine.clamp(-1,1))[mask].double().sum().item();pixels+=mask.sum().item()
    return s/n,math.degrees(angles/max(pixels,1))


def train(a):
    p=plan(a);out=a.runtime/a.arm;out.mkdir(exist_ok=True)
    with lock(out/'run.lock'):
        device=a.device
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
        configure(p['seed'],device);m=model(a.arm).to(device);opt=torch.optim.Adam(m.parameters(),lr=p['lr'])
        gen=torch.Generator().manual_seed(p['seed']);fingerprint=hashlib.sha256(json.dumps({'plan':p,'arm':a.arm},sort_keys=True).encode()).hexdigest()
        epoch=0;best=float('inf')
        if (out/'latest.pt').exists():
            ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
            if ck['fingerprint']!=fingerprint:raise ValueError('Resume fingerprint mismatch')
            m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);epoch=ck['epoch'];best=ck['best_mse']
            gen.set_state(ck['loader_rng']);torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        tr=Cached(a.runtime/'cache/train');va=Cached(a.runtime/'cache/val')
        loader=DataLoader(tr,batch_size=32,shuffle=True,generator=gen,num_workers=0,pin_memory=device=='cuda')
        vl=DataLoader(va,batch_size=4,shuffle=False,num_workers=0,pin_memory=device=='cuda')
        for ep in range(epoch,p['epochs']):
            for idx,total,seen,started in run_epoch(m,opt,loader,device):
                if idx%20==0:
                    write(out/'status.json',{'pid':os.getpid(),'state':'training','arm':a.arm,'epoch':ep+1,'epochs':p['epochs'],'batch':idx+1,'batches':len(loader),'train_mse':total/seen,'patches_per_second':seen/max(time.monotonic()-started,1e-6),'updated':time.time(),'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2 if device=='cuda'else 0})
            write(out/'status.json',{'pid':os.getpid(),'state':'validation','arm':a.arm,'epoch':ep+1,'updated':time.time()})
            vmse,sam=validate(m,vl,device)
            if not math.isfinite(vmse):raise FloatingPointError('Nonfinite validation')
            improved=vmse<best;best=min(best,vmse)
            row={'epoch':ep+1,'train_mse':total/seen,'val_mse':vmse,'val_sam':sam,'val_global_psnr':-10*math.log10(max(vmse,1e-15)),'best_mse':best,'seconds':time.monotonic()-started}
            st={'config':{'plan':p,'sensor':'QB','k':4,'seed':42,'variant':a.arm},'fingerprint':fingerprint,'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep+1,'best_mse':best,'validation_mse':vmse,'loader_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()if device=='cuda'else [],'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            if improved:save(out/'best.pt',st)
            save(out/'latest.pt',st)
            hp=out/'history.json';history=json.loads(hp.read_text())if hp.exists()else [];write(hp,[x for x in history if x['epoch']<ep+1]+[row]);print(json.dumps(row),flush=True)
        write(out/'complete.json',{'arm':a.arm,'epochs':p['epochs'],'best_mse':best,'best_sha256':sha(out/'best.pt'),'latest_sha256':sha(out/'latest.pt'),'fingerprint':fingerprint,'completed':time.time()})
        write(out/'status.json',{'pid':os.getpid(),'state':'complete','epoch':p['epochs'],'updated':time.time()})


def probe(a):
    p=plan(a);configure(42,'cuda');m=model(a.arm).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    ds=Cached(a.runtime/'cache/train');loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
    ready=a.runtime/'benchmark'/f'{a.slots}_{a.arm}.ready';ready.touch()
    go=a.runtime/'benchmark'/f'{a.slots}.go';deadline=time.monotonic()+120
    while not go.exists():
        if time.monotonic()>deadline:raise TimeoutError('Probe barrier')
        time.sleep(.05)
    start=None;count=0
    for idx,total,seen,tick in run_epoch(m,opt,loader,'cuda',limit=40):
        if idx==7:torch.cuda.synchronize();start=time.monotonic()
        elif idx>=8:count+=32
    torch.cuda.synchronize();seconds=time.monotonic()-start
    write(a.runtime/'benchmark'/f'{a.slots}_{a.arm}.json',{'arm':a.arm,'slots':a.slots,'seconds':seconds,'patches':count,'patches_per_second':count/seconds,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'probe_discarded':True,'checkpoint_saved':False})


def benchmark(a):
    plan(a);folder=a.runtime/'benchmark';folder.mkdir(exist_ok=False);rows=[]
    for slots in (1,2,3):
        active=[];started=time.monotonic()
        for arm in ARMS[:slots]:
            log=(folder/f'{slots}_{arm}.log').open('w');p=subprocess.Popen([sys.executable,__file__,'probe','--runtime',str(a.runtime),'--arm',arm,'--slots',str(slots)],stdout=log,stderr=subprocess.STDOUT);active.append((arm,p,log))
        deadline=time.monotonic()+120
        while not all((folder/f'{slots}_{arm}.ready').exists()for arm,_,_ in active):
            if any(p.poll()is not None for _,p,_ in active)or time.monotonic()>deadline:raise RuntimeError('Probe initialization failed')
            time.sleep(.1)
        (folder/f'{slots}.go').touch();codes=[]
        for arm,p,log in active:codes.append(p.wait());log.close()
        if any(codes):
            rows.append({'slots':slots,'safe':False,'exitcodes':codes});break
        probes=[json.loads((folder/f'{slots}_{arm}.json').read_text())for arm,_,_ in active]
        memory=sum(x['peak_reserved_mib']for x in probes)
        rows.append({'slots':slots,'safe':memory<=6000,'summed_peak_reserved_mib':memory,'aggregate_patches_per_second':sum(x['patches_per_second']for x in probes),'probes':probes})
        print(json.dumps(rows[-1]),flush=True)
    safe=[r for r in rows if r['safe']]
    if not safe:raise RuntimeError('No safe GPU concurrency')
    peak=max(r['aggregate_patches_per_second']for r in safe)
    # Prefer fewer contexts unless extra concurrency improves throughput by >5%.
    chosen=min((r for r in safe if r['aggregate_patches_per_second']>=peak*.95),key=lambda r:r['slots'])
    write(a.runtime/'benchmark.json',{'rows':rows,'selected_slots':chosen['slots'],'rule':'within 5% of fastest measured aggregate throughput; smallest safe concurrency; sum reserved <=6000 MiB','precision':'FP32','effective_batch':32,'micro_batch':4,'probe_only_no_retained_weights':True})


def controller(a):
    p=plan(a);slots=json.loads((a.runtime/'benchmark.json').read_text())['selected_slots']
    with lock(a.runtime/'controller.lock'):
        write(a.runtime/'controller.json',{'pid':os.getpid(),'started':time.time(),'slots':slots})
        active={};failed=[]
        while True:
            for arm,(proc,log)in list(active.items()):
                if proc.poll()is None:continue
                log.close();del active[arm]
                if proc.returncode or not(a.runtime/arm/'complete.json').exists():
                    failed.append(arm);write(a.runtime/arm/'failed.json',{'exitcode':proc.returncode,'time':time.time()})
            done=[v for v in ARMS if(a.runtime/v/'complete.json').exists()]
            pending=[v for v in ARMS if v not in active and v not in done and v not in failed]
            if not failed:
                for arm in pending[:max(0,slots-len(active))]:
                    out=a.runtime/arm;out.mkdir(exist_ok=True)
                    if(out/'status.json').exists():
                        old=json.loads((out/'status.json').read_text())
                        if alive(old.get('pid'))and old.get('state')!='complete':raise RuntimeError('Surviving child; duplicate launch refused')
                    log=(out/'train.log').open('a',encoding='utf-8');proc=subprocess.Popen([sys.executable,__file__,'train','--runtime',str(a.runtime),'--arm',arm],stdout=log,stderr=subprocess.STDOUT);active[arm]=(proc,log)
            write(a.runtime/'queue.json',{'controller_pid':os.getpid(),'active':{v:x[0].pid for v,x in active.items()},'completed':done,'failed':failed,'pending':pending,'slots':slots,'updated':time.time()})
            if not active:
                if failed:raise RuntimeError('Failure recorded; other runs were not interrupted')
                write(a.runtime/'training_complete.json',{'arms':list(ARMS),'epochs':100,'evaluation_complete':False,'time':time.time()});return
            time.sleep(5)


def start(a):
    plan(a)
    if os.name!='nt':raise RuntimeError('Detached launcher requires Windows')
    if not(a.runtime/'benchmark.json').exists():raise RuntimeError('Benchmark concurrency first')
    if(a.runtime/'controller.json').exists()and alive(json.loads((a.runtime/'controller.json').read_text()).get('pid')):raise RuntimeError('Controller already active')
    command=subprocess.list2cmdline([sys.executable,__file__,'controller','--runtime',str(a.runtime)])
    cmd='cmd.exe /d /c "'+command+' >> "'+str(a.runtime/'controller.log')+'" 2>&1"'
    ps="$ProgressPreference='SilentlyContinue'; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+cmd.replace("'","''")+"'} | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],capture_output=True,text=True,check=True)
    d=json.loads(r.stdout)
    if d['ReturnValue']!=0:raise RuntimeError(d)
    write(a.runtime/'launch.json',{'wmi':d,'command':command,'time':time.time()});print(json.dumps(d),flush=True)


def status(a):
    for path in [a.runtime/'queue.json']+list(a.runtime.glob('*/status.json')):
        if path.exists():print(str(path),path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','probe','benchmark','train','controller','start','status']);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--data',type=Path);parser.add_argument('--arm',choices=ARMS,default='B0');parser.add_argument('--slots',type=int,default=1);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=parser.parse_args();a.runtime.mkdir(parents=True,exist_ok=True);globals()[a.mode](a)


if __name__=='__main__':main()
