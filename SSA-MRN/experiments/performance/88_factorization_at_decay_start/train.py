"""88 training: the 65 recipe (the 58 C2MS trainer with the five L1 replacements of 65/79/83/84, original-pair L1 + 1.0 x coarse-pair L1 +
0.46 x spectral loss, AMP, selection by original-scale validation MSE including epoch 0, EMA recorded only) with the 84 trunk/branch mechanism
and factorized starts.

Replacements, each matching exactly once in the 58 trainer:
1. the cosine learning rate over E epochs becomes schedule(cfg, arm, epoch): TRUNK keeps lr_start (1e-4); FULL, K2D and K2F decay from
   lr_peak to lr_end with the 65 cosine formula over their decay epochs (FULL: epochs 120-149 of the trunk counter; K2D: 0-29; K2F: 0-9);
2. after the TRUNK epoch listed in cfg['snapshots'] (120) the trainer copies latest.pt, best.pt and best_ema.pt to
   runs/<sensor>_TRUNK_s<seed>/snapshots/e120/ with their hashes (84);
3. make_model loads the factorized initial weights of K2D/K2F (runtime/init/<sensor>_<arm>_s<seed>.pt, written by `run.py --mode factorize`)
   and checks the file hash against its record; in the benchmark and preflight the K2 structure keeps its seeded initialization;
4. the preflight initial-state check (4-band models equal to 43) is applied to the full-size arms only (check_initial).

FULL starts from the TRUNK snapshot (start_branch, as 84 B150): model, Adam, GradScaler, RNG, data-order generator, history and the best-so-far
continue. K2D and K2F are new runs (new Adam, data order from the seed) whose initial weights are the factorized ones."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,shutil,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
L1_REPLACEMENTS=(  # identical to 65/79/83/84 train.py
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)
SCHEDULE_REPLACEMENTS=(  # identical to 84 train.py
    ("lr=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*epoch/(E-1)))","lr=schedule(cfg,arm,epoch)"),
    ("history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();",
     "history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();snapshot_hook(cfg,arm,epoch,cp);"),
)
INIT_REPLACEMENTS=(  # as 87 train.py
    ("    m=Model(arm,SENSORS[sensor]['bands'],seed).cuda()\n","    m=load_initial(Model(arm,SENSORS[sensor]['bands'],seed),sensor,arm,seed).cuda()\n"),
    ("if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
     "check_initial(sd,h,sensor,arm,seed)"),
)
_STATE={'mode':None}


def schedule(cfg,arm,epoch):
    """Learning rate of zero-based epoch `epoch` of the run's own counter."""
    if arm=='TRUNK':return cfg['lr_start']
    k=int(cfg['decay_start'][arm]);d=int(cfg['epochs'][arm])-k;j=epoch-k;peak=float(cfg['lr_peak'][arm])
    if j<0:return cfg['lr_start']  # never reached: FULL starts at epoch 120, K2D/K2F at 0
    if d<2:return cfg['lr_end']
    return cfg['lr_end']+.5*(peak-cfg['lr_end'])*(1+math.cos(math.pi*j/(d-1)))


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


def snapshot_files(cfg,sensor,seed):
    """Hash-checked TRUNK epoch-120 snapshot folder (the shared start of FULL and K2D)."""
    k=int(cfg['decay_start']['FULL']);src=snapshot_dir(cfg,sensor,seed,k);meta=json.loads((src/'snapshot.json').read_text(encoding='utf8'))
    for n,e in meta['files'].items():
        if T.sha(src/n)!=e['sha256']:raise RuntimeError('snapshot hash mismatch '+str(src/n))
    return k,src,meta


def start_branch(cfg,sensor,arm,seed):
    """FULL: copy the TRUNK snapshot into the branch run once (hashes checked); later restarts resume the branch's own latest.pt (84)."""
    import torch
    out=Path(cfg['runtime'])/'runs'/T.run_name(sensor,arm,seed);cp=out/'checkpoints';marker=out/'branch_init.json'
    if marker.exists():return
    k,src,meta=snapshot_files(cfg,sensor,seed)
    ck=torch.load(src/'latest.pt',map_location='cpu',weights_only=False)
    if ck['epoch']!=k or ck['next_batch']!=0 or ck['order'] is not None or ck['fingerprint']!=cfg['fingerprint'] or ck['seed']!=seed:raise RuntimeError('snapshot state differs from a clean epoch boundary '+str(src))
    if cp.exists():shutil.rmtree(cp)  # only snapshot copies can be here: training never starts before branch_init.json is written
    cp.mkdir(parents=True)
    for n in ('latest.pt','best.pt','best_ema.pt'):shutil.copy2(src/n,cp/n)
    T.write(out/'history.json',ck['history'])
    T.write(marker,dict(time_kst=T.now(),trunk_epoch=k,decay_epochs=int(cfg['epochs'][arm])-k,total_epochs=int(cfg['epochs'][arm]),snapshot=str(src),snapshot_files=meta['files'],trunk_best_epoch=ck['best_epoch']))


def init_path(cfg,sensor,arm,seed):return Path(cfg['runtime'])/'init'/f'{sensor}_{arm}_s{seed}.pt'


def load_initial(m,sensor,arm,seed):
    """K2D/K2F: factorized weights whose hash equals their record; full-size arms and the benchmark/preflight keep the seeded initialization."""
    import model as M
    if arm not in M.FACTORIZED or _STATE['mode'] in ('bench','preflight'):return m
    import torch
    cfg=_STATE['cfg'];p=init_path(cfg,sensor,arm,seed);rec=p.with_suffix('.json')
    if not rec.exists() or T.sha(p)!=json.loads(rec.read_text(encoding='utf8'))['sha256']:raise RuntimeError('factorized initial weights missing or not matching their record: '+str(p))
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True);return m


def check_initial(sd,h,sensor,arm,seed):
    """Full-size arms: the 4-band initial state equals 43 C2 of the same seed (as 58/84). K2 arms start from factorized weights (hash-checked at load)."""
    import model as M
    if arm in M.FULL_SIZE and T.SENSORS[sensor]['bands']==4 and h!=T.state_hash(M.M43.Model(M.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS+SCHEDULE_REPLACEMENTS+INIT_REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('58 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_l1s_88');mod.__file__=str(SOURCE)
    mod.schedule=schedule;mod.snapshot_hook=snapshot_hook;mod.load_initial=load_initial;mod.check_initial=check_initial
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
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));_STATE.update(cfg=cfg,mode=a.mode)
    try:
        if a.mode=='bench':T.benchmark(cfg,a.sensor,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        elif a.arm=='FULL':start_branch(cfg,a.sensor,a.arm,a.seed);T.train(cfg,a.sensor,a.arm,a.seed,True)
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
