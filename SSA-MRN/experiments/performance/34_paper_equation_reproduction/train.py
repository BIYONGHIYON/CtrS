"""Fresh QB 100-epoch paper-equation run, epoch-boundary resumable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, json, math, random, shutil, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from model import PaperSSAMRN


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)


def save(p,d):
    tmp=p.with_suffix('.tmp')
    with open(tmp,'wb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def run(a):
    torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    config=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    r=a.runtime;r.mkdir(parents=True,exist_ok=True)
    model=PaperSSAMRN().cuda();opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    src={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};src['config.json']=sha(Path(__file__).with_name('config.json'))
    planfile=r/'plan.json'
    if planfile.exists():
        plan=json.loads(planfile.read_text(encoding='utf-8'));assert plan['source_hashes']==src,'Source changed'
    else:
        if a.resume:raise FileNotFoundError('No plan to resume')
        data={}
        for split,filename in [('train','train_qb.h5'),('val','valid_qb.h5')]:
            paths=list(a.data.rglob(filename));assert len(paths)==1
            with h5py.File(paths[0],'r') as f:data[split]={'path':str(paths[0]),'sha256':sha(paths[0]),'shapes':{k:list(f[k].shape) for k in ('pan','lms','ms','gt')}}
        assert data['train']['shapes']['pan']==[17139,1,64,64] and data['val']['shapes']['pan']==[1905,1,64,64]
        plan={'config':config,'source_hashes':src,'data':data,'environment':{'torch':torch.__version__,'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'examples_frozen_before_test':[1,8,11,13,19],'determinism':'warn_only for CUDA bicubic backward; not bitwise guaranteed','created_utc':time.time()}
        write(planfile,plan)
        for p in Path(__file__).parent.iterdir():
            if p.suffix in ('.py','.json'):
                dest=r/'provenance/source'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('loading original train/validation H5 into CPU RAM',flush=True)
    datasets={}
    for split,meta in plan['data'].items():
        assert sha(meta['path'])==meta['sha256']
        with h5py.File(meta['path'],'r') as f:datasets[split]=[torch.from_numpy(np.asarray(f[k],dtype=np.float32)/2047.) for k in ('pan','lms','ms','gt')]
        assert all(torch.isfinite(x).all() for x in datasets[split])
    start=1;best=float('inf');history=[];cpdir=r/'checkpoints/PAPER_QB_s42';cpdir.mkdir(parents=True,exist_ok=True)
    if a.resume:
        ck=torch.load(cpdir/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck['best_validation_mse'];history=ck['history']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    elif (cpdir/'latest.pt').exists():raise FileExistsError('Use --resume explicitly')
    def validation():
        model.eval();total=0.;sam=0.;psnr=0.;count=0
        with torch.inference_mode():
            for off in range(0,len(datasets['val'][0]),32):
                p,l,m,g=[x[off:off+32].cuda() for x in datasets['val']];y=model(p,l,m)
                if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite validation')
                total+=float((y-g).square().sum(dtype=torch.float64));count+=g.numel()
                err=(y-g).square().mean((2,3));psnr+=float((-10*err.clamp_min(1e-12).log10()).mean(1).sum())
                norm=torch.linalg.vector_norm(y,dim=1)*torch.linalg.vector_norm(g,dim=1);valid=norm>1e-12
                angle=torch.acos(((y*g).sum(1)/norm.clamp_min(1e-12)).clamp(-1,1))*180/math.pi
                sam+=sum(float(angle[i][valid[i]].mean()) if valid[i].any() else 0. for i in range(len(g)))
        n=len(datasets['val'][0]);return total/count,sam/n,psnr/n
    initial=validation();write(r/'initial_validation.json',dict(zip(('mse','sam','psnr'),initial)));print('initial_validation',initial,flush=True)
    for epoch in range(start,config['epochs']+1):
        model.train();begin=time.monotonic();total=0.;count=0;order=torch.randperm(len(datasets['train'][0]),generator=torch.Generator().manual_seed(42+epoch));batchhash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        for step,off in enumerate(range(0,len(order),32),1):
            ids=order[off:off+32];p,l,m,g=[x[ids].cuda() for x in datasets['train']];opt.zero_grad(set_to_none=True);y=model(p,l,m);loss=(y-g).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite gradient')
            opt.step();total+=float(loss.detach())*len(g);count+=len(g)
            if step%100==0:print(f'epoch={epoch} step={step}/536 mse={total/count:.9f}',flush=True)
        val,sam,psnr=validation();improved=val<best
        if improved:best=val
        row={'epoch':epoch,'train_mse':total/count,'val_mse':val,'val_sam':sam,'val_psnr':psnr,'seconds':time.monotonic()-begin,'batch_order_sha256':batchhash};history.append(row)
        ck={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'validation_mse':val,'best_validation_mse':best,'config':config,'history':history,'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        save(cpdir/'latest.pt',ck)
        if improved:save(cpdir/'best.pt',ck)
        write(r/'history.json',history);write(r/'status.json',{'state':'training','epoch':epoch,'best_validation_mse':best,'last':row});print(json.dumps(row),flush=True)
    write(r/'training_complete.json',{'epochs':100,'best_validation_mse':best,'best_sha256':sha(cpdir/'best.pt'),'latest_sha256':sha(cpdir/'latest.pt')})
    # Whole-scene evaluation uses only the frozen validation-selected best.
    import evaluate
    evaluate.run(a)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a)
    except Exception as e:
        import traceback
        write(a.runtime/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
