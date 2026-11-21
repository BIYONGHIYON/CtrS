"""89 training: the 88 trainer (the 58 C2MS trainer with the 65 L1 replacements, AMP, selection by original-scale validation MSE including
epoch 0, EMA recorded only) with per-arm loss weights and schedules:

  W2T        : lr 1e-4 constant for 120 epochs; after epoch 120 latest/best/best_ema are copied to snapshots/e120 with hashes (88 TRUNK)
  W2K        : factorized W2T epoch-120 weights (hash-checked file written by `run.py --mode factorize`), new Adam, 30-epoch cosine
               1e-4 -> 1e-6 (88 K2D)
  L<lam>C<c> : 50 epochs, epochs 0-39 at 1e-4, epochs 40-49 cosine 1e-4 -> 1e-6 (88 schedule shortened); loss L1 + c x coarse L1 +
               lam x spectral loss

The loss weights come from the arm name (model.parse), so the stage-B arms (lambda chosen after stage A) need no config change. The
W2 output implementation is the sealed cfg['w2_out_impl']."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,shutil,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
L1_REPLACEMENTS=(  # identical to 65/79/83/84/88 train.py
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)
SCHEDULE_REPLACEMENTS=(  # as 88
    ("lr=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*epoch/(E-1)))","lr=schedule(cfg,arm,epoch)"),
    ("history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();",
     "history[-1]['epoch_seconds']=duration;write(out/'history.json',history);save();last_save=time.monotonic();snapshot_hook(cfg,arm,epoch,cp);"),
)
INIT_REPLACEMENTS=(  # as 88
    ("    m=Model(arm,SENSORS[sensor]['bands'],seed).cuda()\n","    m=load_initial(Model(arm,SENSORS[sensor]['bands'],seed),sensor,arm,seed).cuda()\n"),
    ("if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
     "check_initial(sd,h,sensor,arm,seed)"),
)
_STATE={'mode':None}
TRUNK_EPOCHS,K2D_EPOCHS,SCREEN_EPOCHS,SCREEN_CONSTANT=120,30,50,40


def epochs(cfg,arm):
    import model as M
    if arm=='W2T':return TRUNK_EPOCHS
    if arm=='W2K':return K2D_EPOCHS
    if M.is_screen(arm):return SCREEN_EPOCHS
    raise ValueError(arm)


def spectral_weight(cfg,arm):
    import model as M
    return M.parse(arm)[0]


def coarse_weight(cfg,arm):
    import model as M
    return M.parse(arm)[1]


def cosine(cfg,j,d,peak):
    if d<2:return cfg['lr_end']
    return cfg['lr_end']+.5*(peak-cfg['lr_end'])*(1+math.cos(math.pi*j/(d-1)))


def schedule(cfg,arm,epoch):
    """Learning rate of zero-based epoch `epoch` of the run's own counter."""
    import model as M
    if arm=='W2T':return cfg['lr_start']
    if arm=='W2K':return cosine(cfg,epoch,K2D_EPOCHS,cfg['lr_start'])
    if M.is_screen(arm):return cfg['lr_start'] if epoch<SCREEN_CONSTANT else cosine(cfg,epoch-SCREEN_CONSTANT,SCREEN_EPOCHS-SCREEN_CONSTANT,cfg['lr_start'])
    raise ValueError(arm)


def snapshot_dir(cfg,sensor,seed,epoch=TRUNK_EPOCHS):return Path(cfg['runtime'])/'runs'/f'{sensor}_W2T_s{seed}'/'snapshots'/f'e{epoch}'


def snapshot_hook(cfg,arm,epoch,cp):
    if arm!='W2T' or epoch!=TRUNK_EPOCHS:return
    dest=cp.parent/'snapshots'/f'e{epoch}';tmp=dest.with_name(dest.name+'_building')
    if dest.exists():return
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in ('latest.pt','best.pt','best_ema.pt'):shutil.copy2(cp/name,tmp/name)
    T.write(tmp/'snapshot.json',dict(time_kst=T.now(),epoch=epoch,files={n:dict(sha256=T.sha(tmp/n),bytes=(tmp/n).stat().st_size) for n in ('latest.pt','best.pt','best_ema.pt')}))
    os.replace(tmp,dest)


def snapshot_files(cfg,sensor,seed):
    src=snapshot_dir(cfg,sensor,seed);meta=json.loads((src/'snapshot.json').read_text(encoding='utf8'))
    for n,e in meta['files'].items():
        if T.sha(src/n)!=e['sha256']:raise RuntimeError('snapshot hash mismatch '+str(src/n))
    return TRUNK_EPOCHS,src,meta


def init_path(cfg,sensor,arm,seed):return Path(cfg['runtime'])/'init'/f'{sensor}_{arm}_s{seed}.pt'


def load_initial(m,sensor,arm,seed):
    import model as M
    if arm not in M.FACTORIZED or _STATE['mode'] in ('bench','preflight'):return m
    import torch
    cfg=_STATE['cfg'];p=init_path(cfg,sensor,arm,seed);rec=p.with_suffix('.json')
    if not rec.exists() or T.sha(p)!=json.loads(rec.read_text(encoding='utf8'))['sha256']:raise RuntimeError('factorized initial weights missing or not matching their record: '+str(p))
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True);return m


def check_initial(sd,h,sensor,arm,seed):
    """Full-size 2x2 arms on 4-band sensors: the initial state equals 43 C2 of the same seed converted by to_2x2 (same seeded weights)."""
    import model as M
    if arm in M.FACTORIZED or T.SENSORS[sensor]['bands']!=4:return
    ref=M.to_2x2(M.M54.Model('C2MS',4,seed))
    c2=M.M43.Model('C2',seed).state_dict();base=M.M54.Model('C2MS',4,seed).state_dict()
    if T.state_hash(c2)!=T.state_hash(base):raise AssertionError('54 C2MS initial state differs from 43 C2')
    if h!=T.state_hash(ref.state_dict()):raise AssertionError('2x2 initial state differs from the converted 43 C2 '+arm)


def set_impl(cfg):
    import model as M
    M.OUT_IMPL['value']=cfg.get('w2_out_impl','convT')


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS+SCHEDULE_REPLACEMENTS+INIT_REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('58 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_l1s_89');mod.__file__=str(SOURCE)
    mod.schedule=schedule;mod.snapshot_hook=snapshot_hook;mod.load_initial=load_initial;mod.check_initial=check_initial
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__)
    mod.epochs=epochs;mod.spectral_weight=spectral_weight;mod.weight=coarse_weight  # per-arm loss weights and lengths from the arm name
    return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layout',default='channels_last');p.add_argument('--tag',default='single');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));_STATE.update(cfg=cfg,mode=a.mode);set_impl(cfg)
    try:
        if a.mode=='bench':T.benchmark(cfg,a.sensor,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
