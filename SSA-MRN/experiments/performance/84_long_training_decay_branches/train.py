"""C2MS-SP training (the 65 recipe: the 58 C2MS trainer with the five L1 replacements of 65/79/83, original-pair L1 + 1.0 x coarse-pair
L1 + 0.46 x spectral loss, AMP, selection by original-scale validation MSE including epoch 0, EMA recorded only) with a longer schedule.

Two more replacements, each matching exactly once:
1. the cosine learning rate over E epochs becomes schedule(cfg, arm, epoch): TRUNK keeps lr_start (1e-4); branch B<N> starts from the
   TRUNK state after 0.8 N epochs and decays 1e-4 -> 1e-6 over the last D = 0.2 N epochs with the same cosine formula as 65
   (lr = 1e-6 + 0.5 (1e-4 - 1e-6)(1 + cos(pi j / (D - 1))), j = 0 .. D - 1);
2. after every finished TRUNK epoch listed in cfg['snapshots'] (120, 240, 480, 800) the trainer copies latest.pt, best.pt and best_ema.pt to
   runs/<sensor>_TRUNK_s<seed>/snapshots/e<epoch>/ (same process, at the epoch boundary, so the three files are consistent).

start_branch() seeds a branch run with the TRUNK snapshot (its latest.pt becomes the branch latest.pt; best/best_ema carry the selection
over the trunk epochs) and the branch is then trained by the unchanged resume path: model, Adam, GradScaler, RNG, data-order generator,
history and the best-so-far all continue, so a branch is the run an N-epoch training with this schedule would have produced. Selection of
B<N> is the validation-MSE minimum over epochs 0 .. N of its own path."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,shutil,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
L1_REPLACEMENTS=(  # identical to 65/79/83 train.py
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)
SCHEDULE_REPLACEMENTS=(
    ("lr=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*epoch/(E-1)))","lr=schedule(cfg,arm,epoch)"),
    ("history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();",
     "history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();snapshot_hook(cfg,arm,epoch,cp);"),
)


def branch_start(cfg,arm):
    """Trunk epoch a branch starts from (0.8 N) and its decay length (0.2 N)."""
    n=int(cfg['epochs'][arm]);k=int(round(cfg['branch_fraction']*n));return k,n-k


def schedule(cfg,arm,epoch):
    """Learning rate of zero-based epoch `epoch`."""
    if arm=='TRUNK':return cfg['lr_start']
    k,d=branch_start(cfg,arm)
    if epoch<k:return cfg['lr_start']  # never reached in a branch run (it starts at epoch k); equal to the trunk
    j=epoch-k
    if d<2:return cfg['lr_end']
    return cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*j/(d-1)))


def snapshot_dir(cfg,sensor,seed,epoch):return Path(cfg['runtime'])/'runs'/f'{sensor}_TRUNK_s{seed}'/'snapshots'/f'e{epoch}'


def snapshot_hook(cfg,arm,epoch,cp):
    if arm!='TRUNK' or epoch not in cfg['snapshots']:return
    dest=cp.parent/'snapshots'/f'e{epoch}';tmp=dest.with_name(dest.name+'_building')
    if dest.exists():return
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in ('latest.pt','best.pt','best_ema.pt'):shutil.copy2(cp/name,tmp/name)
    T.write(tmp/'snapshot.json',dict(time_kst=T.now(),epoch=epoch,files={n:dict(sha256=T.sha(tmp/n),bytes=(tmp/n).stat().st_size) for n in ('latest.pt','best.pt','best_ema.pt')}))
    os.replace(tmp,dest)


def start_branch(cfg,sensor,arm,seed):
    """Copy the TRUNK snapshot at 0.8 N into the branch run once (hashes checked); later restarts resume the branch's own latest.pt."""
    import torch
    out=Path(cfg['runtime'])/'runs'/T.run_name(sensor,arm,seed);cp=out/'checkpoints';marker=out/'branch_init.json'
    if marker.exists():return
    k,d=branch_start(cfg,arm);src=snapshot_dir(cfg,sensor,seed,k);meta=json.loads((src/'snapshot.json').read_text(encoding='utf8'))
    for n,e in meta['files'].items():
        if T.sha(src/n)!=e['sha256']:raise RuntimeError('snapshot hash mismatch '+str(src/n))
    ck=torch.load(src/'latest.pt',map_location='cpu',weights_only=False)
    if ck['epoch']!=k or ck['next_batch']!=0 or ck['order'] is not None or ck['fingerprint']!=cfg['fingerprint'] or ck['seed']!=seed:raise RuntimeError('snapshot state differs from a clean epoch boundary '+str(src))
    if cp.exists():shutil.rmtree(cp)  # only snapshot copies can be here: training never starts before branch_init.json is written
    cp.mkdir(parents=True)
    for n in ('latest.pt','best.pt','best_ema.pt'):shutil.copy2(src/n,cp/n)
    T.write(out/'history.json',ck['history'])
    T.write(marker,dict(time_kst=T.now(),trunk_epoch=k,decay_epochs=d,total_epochs=k+d,snapshot=str(src),snapshot_files=meta['files'],trunk_best_epoch=ck['best_epoch']))


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS+SCHEDULE_REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('58 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_l1s_long');mod.__file__=str(SOURCE);mod.schedule=schedule;mod.snapshot_hook=snapshot_hook
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):return int(cfg['epochs'][arm])
if __name__=='__main__':
    import argparse
    from model import ARMS
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layout',default='channels_last');p.add_argument('--tag',default='single');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    try:
        if a.mode=='bench':T.benchmark(cfg,a.sensor,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        elif a.arm=='TRUNK':T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
        else:start_branch(cfg,a.sensor,a.arm,a.seed);T.train(cfg,a.sensor,a.arm,a.seed,True)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
