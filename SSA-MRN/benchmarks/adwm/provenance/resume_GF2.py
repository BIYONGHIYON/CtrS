"""Resume the interrupted ADWM GF2 run (stopped at epoch 485 by a disk-quota write failure on 2026-11-24 02:11) from its epoch-484
latest.pt for epochs 485-500. The official benchmarks/adwm/train.py is not modified: this file repeats its setup and per-epoch loop
exactly (same model, optimizer, StepLR, DataLoader, loss, saving) and restores model, optimizer, scheduler, torch/cuda RNG and history.
The DataLoader shuffle draws from the torch global RNG, which is restored, so the epoch order continues as in an uninterrupted run."""
import json,sys,datetime
from pathlib import Path
SRC=Path.home()/'ctrs-adwm-train/SSA-MRN/benchmarks/adwm';sys.path.insert(0,str(SRC))
import numpy as np,torch,random
from torch import nn
from torch.utils.data import DataLoader
import train as A
from upstream.model import bwnet_lagconv_sota as official
data_root=Path('/home/gpu_01/datasets/.PAN-MS-direct-20261123');out=Path.home()/'adwm-training-vFKc5M/gf2'
folder,scale,epochs,train_count,valid_count=A.SENSORS['gf2'];data=data_root/folder/'Training Dataset'
random.seed(A.SEED);np.random.seed(A.SEED);torch.manual_seed(A.SEED)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda:0')
model=official.BWNet(ms_dim=4).to(device)
optimizer=torch.optim.Adam([{'params':model.parameters(),'initial_lr':0.002}],lr=0.002,weight_decay=0)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.5,last_epoch=0)
criterion=nn.L1Loss()
training=DataLoader(A.Patches(data/'train_gf2.h5',train_count,scale),batch_size=A.BATCH,shuffle=True,drop_last=True)
validation=DataLoader(A.Patches(data/'valid_gf2.h5',valid_count,scale),batch_size=A.BATCH,shuffle=False)
ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
history=ck['history'];done=history[-1]['epoch']
lines=[json.loads(l) for l in (out/'history.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
if done!=484 or lines[-1]!=history[-1] or len(history)!=done:raise RuntimeError(f'unexpected state: latest {done}, jsonl {lines[-1]}')
model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'],device)
best=min(r['valid_l1'] for r in history)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
(out.parent/'gf2_resume.json').write_text(json.dumps(dict(time_kst=now,reason='disk quota exceeded while saving latest.pt at epoch 485 (2026-11-24 02:11), caused by an unrelated cache build in the same account',
    resumed_from_epoch=done,latest_pt_mtime='2026-11-24 02:09:42',best_so_far=best,lr_next=optimizer.param_groups[0]['lr'],script=str(Path(__file__).resolve()),
    note='official train.py unchanged; epochs 485-500 run with the same loop; numpy/python RNG are not used by the loop'),indent=1),encoding='utf-8')
print(f'GF2 resume from epoch {done} lr {optimizer.param_groups[0]["lr"]} best {best}',flush=True)
for epoch in range(done+1,epochs+1):
    model.train();total=0.0;lr=optimizer.param_groups[0]['lr']
    for ms,pan,gt in training:
        optimizer.zero_grad();loss=criterion(model(ms.to(device),pan.to(device)),gt.to(device));loss.backward();optimizer.step();total+=loss.item()
    scheduler.step()
    model.eval();valid_total=0.0
    with torch.inference_mode():
        for ms,pan,gt in validation:
            prediction=model(ms.to(device),pan.to(device));valid_total+=criterion(prediction,gt.to(device)).item()*len(gt)
    valid_l1=valid_total/valid_count
    if not np.isfinite(valid_l1) or not np.isfinite(total):raise RuntimeError(f'Non-finite loss at epoch {epoch}')
    row=dict(epoch=epoch,lr=lr,train_l1=total/len(training),valid_l1=valid_l1);history.append(row)
    if valid_l1<best:best=valid_l1;A.save_atomic(out/'best.pt',{'model':model.state_dict(),**row})
    A.save_atomic(out/'latest.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),'history':history})
    with (out/'history.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(row)+'\n')
    print(f'GF2 {epoch}/{epochs} {row}',flush=True)
