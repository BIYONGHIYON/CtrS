"""Fixed-condition training using the existing verified numerical loop."""
import hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ssai_suite as suite
from .model import STAGES


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def common_state(m):
    return {k:v for k,v in m.state_dict().items() if not k.endswith(('.gamma', '.pan_response'))}


def train(a):
    p = suite.plan(a); out = a.runtime/a.arm; out.mkdir(exist_ok=True)
    with suite.lock(out/'run.lock'):
        if (out/'latest.pt').exists(): raise RuntimeError('Automatic restart refused')
        suite.configure(p['seed'], a.device)
        m = suite.model(a.arm).to(a.device); m.set_response(p['pan_response'])
        if state_hash(common_state(m)) != p['initial_common_sha256']: raise ValueError('Common initialization mismatch')
        initial = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        opt = torch.optim.Adam(m.parameters(), lr=p['lr'])
        def finite_step(optimizer, args, kwargs):
            gradients = [x.grad for x in m.parameters() if x.grad is not None]
            if not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all().item():
                raise FloatingPointError('Nonfinite gradient; update refused')
        opt.register_step_pre_hook(finite_step)
        gen = torch.Generator().manual_seed(p['seed'])
        tr, va = (suite.Cached(Path(p['cache_root'])/split) for split in ('train','val'))
        loader = suite.make_loader(tr, 'gpu_partial', 32, gen, a.device)
        vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0, pin_memory=a.device=='cuda')
        best = float('inf'); history = []
        try:
            for epoch in range(1,101):
                for idx,total,seen,started in suite.run_epoch(m,opt,loader,a.device,micro=32):
                    if (a.runtime/'stop.request').exists():
                        suite.write(out/'status.json', {'pid':os.getpid(),'state':'stopped','epoch':epoch,'updated':time.time()}); return
                    if idx%20==0:
                        suite.write(out/'status.json', {'pid':os.getpid(),'state':'training','arm':a.arm,'epoch':epoch,'epochs':100,'batch':idx+1,'batches':len(loader),'train_mse':total/seen,'patches_per_second':seen/max(time.monotonic()-started,1e-6),'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'updated':time.time()})
                suite.write(out/'status.json', {'pid':os.getpid(),'state':'validation','epoch':epoch,'updated':time.time()})
                vmse,sam = suite.validate(m,vl,a.device)
                if not math.isfinite(vmse) or not math.isfinite(sam): raise FloatingPointError('Nonfinite validation')
                if epoch==1 and vmse>=p['divergence']['first_epoch_mse_limit']:
                    raise FloatingPointError('Epoch1 validation exceeds fixed 2x A0 epoch1 limit')
                improved=vmse<best; best=min(best,vmse)
                row={'epoch':epoch,'train_mse':total/seen,'val_mse':vmse,'val_sam':sam,'val_global_psnr':-10*math.log10(max(vmse,1e-15)),'best_mse':best,'seconds':time.monotonic()-started}
                row['gamma']={f'{s}.{i}':float(b.gamma.detach().cpu()) for s in STAGES for i,b in enumerate(getattr(m,s))}
                changes={}
                current=m.state_dict()
                for stage in STAGES:
                    keys=[k for k in initial if k.startswith(stage+'.') and k.endswith('.weight') and '.conv6t12.' not in k]
                    num=sum((current[k].detach().cpu().double()-initial[k].double()).square().sum().item() for k in keys)
                    den=sum(initial[k].double().square().sum().item() for k in keys)
                    changes[stage]=math.sqrt(num/den)
                row['active_convolution_weight_relative_change']=changes
                ck={'config':{'plan':p,'sensor':'QB','k':4,'seed':42,'variant':a.arm},'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_mse':best,'validation_mse':vmse,'loader_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
                if improved: suite.save(out/'best.pt',ck)
                history.append(row); suite.write(out/'history.json',history); suite.save(out/'latest.pt',ck)
                print(json.dumps(row),flush=True)
            suite.write(out/'complete.json',{'arm':a.arm,'epochs':100,'best_mse':best,'best_sha256':suite.sha(out/'best.pt'),'latest_sha256':suite.sha(out/'latest.pt'),'completed':time.time()})
            suite.write(out/'status.json',{'pid':os.getpid(),'state':'complete','epoch':100,'updated':time.time()})
        except Exception as error:
            suite.write(out/'status.json',{'pid':os.getpid(),'state':'failed','reason':str(error),'epoch':epoch,'updated':time.time(),'automatic_restart':False}); raise
