"""Train the public-code architecture from scratch for 150 epochs; detached two-slot queue."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, base64, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common.model import RestoredPansharpeningNet
SENSORS={'QB':4,'GF2':4,'WV3':8}
KEYS=('pan','lms','ms','gt')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8');os.replace(t,p)
def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x
class Cached(Dataset):
    def __init__(self,p):self.p=Path(p);self.arrays=None;self.n=len(np.load(self.p/'pan.npy',mmap_mode='r'))
    def __len__(self):return self.n
    def __getitem__(self,idx):
        if self.arrays is None:self.arrays={k:np.load(self.p/(k+'.npy'),mmap_mode='r') for k in KEYS}
        return tuple(torch.from_numpy(np.array(self.arrays[k][idx],copy=True)) for k in KEYS)
    def __getstate__(self):return dict(p=self.p,n=self.n,arrays=None)
def loader(ds,shuffle,epoch):
    order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(42+epoch)).tolist() if shuffle else list(range(len(ds)))
    batches=[order[i:i+32] for i in range(0,len(order),32)]
    return DataLoader(ds,batch_size=None,sampler=batches,num_workers=1,pin_memory=True,prefetch_factor=1)
def train(a):
    cfg=json.loads((a.runtime/'config.json').read_text());plan=cfg['sensors'][a.sensor]
    for name,digest in cfg['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed: '+name)
    torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    out=a.runtime/a.sensor;out.mkdir(exist_ok=True)
    model=RestoredPansharpeningNet(SENSORS[a.sensor],4).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    current=out/'latest.pt'
    epoch=0;history=[];best=float('inf');best_epoch=0
    if current.exists():
        ck=torch.load(current,map_location='cpu',weights_only=False)
        assert ck['sensor']==a.sensor and ck['channels']==SENSORS[a.sensor]
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        epoch=ck['epoch'];history=ck['history'];best=ck['best_mse'];best_epoch=ck['best_epoch']
        torch.set_rng_state(ck['torch_rng_state'])
    assert 0<=epoch<=150
    ds=Cached(Path(plan['cache'])/'train');val=loader(Cached(Path(plan['cache'])/'val'),False,0)
    def evaluate():
        model.eval();s=torch.zeros((),device='cuda',dtype=torch.float64);n=0
        with torch.inference_mode():
            for batch in val:
                pan,lms,ms,gt=[x.cuda(non_blocking=True) for x in batch]
                diff=model(pan,lms,ms)-gt;s+=diff.square().sum().double();n+=gt.numel()
        value=s.item()/n
        if not math.isfinite(value):raise RuntimeError('Nonfinite validation')
        return value
    def save(name):
        state=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),epoch=epoch,sensor=a.sensor,channels=SENSORS[a.sensor],seed=42,ssai_dimension=4,learning_rate=opt.param_groups[0]['lr'],batch_size=32,best_mse=best,best_epoch=best_epoch,history=history,torch_rng_state=torch.get_rng_state())
        tmp=out/(name+'.tmp');torch.save(state,tmp);os.replace(tmp,out/(name+'.pt'))
    start=time.monotonic()
    while epoch<150:
        for group in opt.param_groups:group["lr"]=1e-4 if epoch<100 else 1e-5
        model.train();total=torch.zeros((),device='cuda',dtype=torch.float64);count=0;last=0
        for idx,batch in enumerate(loader(ds,True,epoch)):
            pan,lms,ms,gt=[x.cuda(non_blocking=True) for x in batch]
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(pan,lms,ms),gt)
            loss.backward();opt.step();total+=loss.detach().double()*gt.numel();count+=gt.numel()
            if time.monotonic()-last>=5:
                write(out/'status.json',dict(state='training',pid=os.getpid(),sensor=a.sensor,epoch=epoch+1,batch=idx+1,batches=math.ceil(len(ds)/32),updated=time.time(),peak_mib=torch.cuda.max_memory_allocated()/1024**2));last=time.monotonic()
        write(out/'status.json',dict(state='validation',pid=os.getpid(),epoch=epoch+1,updated=time.time()))
        mse=evaluate();epoch+=1;train_mse=total.item()/count
        if not math.isfinite(train_mse):raise RuntimeError('Nonfinite training')
        row=dict(epoch=epoch,train_mse=train_mse,val_mse=mse,lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-start)
        history.append(row);improved=mse<best
        if improved:best=mse;best_epoch=epoch;save('best')
        if epoch==100:
            import shutil
            shutil.copy2(out/'best.pt',out/'best_at_100.pt')
        save('latest');write(out/'history.json',history);print(json.dumps(row),flush=True)
    write(out/'complete.json',dict(epoch=epoch,best_epoch=best_epoch,best_mse=best,best_sha256=sha(out/'best.pt'),latest_sha256=sha(out/'latest.pt')))
    write(out/'status.json',dict(state='complete',epoch=epoch,updated=time.time()))
def controller(a):
    pending=list(SENSORS);active={};failed=[]
    while pending or active:
        while pending and len(active)<2:
            sensor=pending.pop(0);out=a.runtime/sensor;out.mkdir(exist_ok=True)
            if (out/'complete.json').exists():continue
            log=(out/'train.log').open('a');p=subprocess.Popen([sys.executable,__file__,'train','--runtime',str(a.runtime),'--sensor',sensor],stdout=log,stderr=subprocess.STDOUT);log.close();active[sensor]=p
        for sensor,p in list(active.items()):
            if p.poll() is not None:
                if p.returncode:failed.append(dict(sensor=sensor,exit_code=p.returncode))
                del active[sensor]
        write(a.runtime/'queue.json',dict(active={s:p.pid for s,p in active.items()},pending=pending,failed=failed,updated=time.time()))
        time.sleep(3)
    if failed:raise RuntimeError(str(failed))
    write(a.runtime/'training_complete.json',dict(sensors=list(SENSORS),epoch=150,updated=time.time(),evaluation='pending'))
def start(a):
    cmd=subprocess.list2cmdline([sys.executable,__file__,'controller','--runtime',str(a.runtime)])
    line='cmd.exe /d /c "'+cmd+' >> "'+str(a.runtime/'controller.log')+'" 2>&1"'
    ps="Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+line.replace("'","''")+"'} | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],capture_output=True,text=True,check=True)
    print(result.stdout);info=json.loads(result.stdout)
    if info['ReturnValue']!=0:raise RuntimeError(info)
    write(a.runtime/'controller.json',info)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','controller','train']);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--sensor',choices=list(SENSORS));a=p.parse_args();globals()[a.command](a)
