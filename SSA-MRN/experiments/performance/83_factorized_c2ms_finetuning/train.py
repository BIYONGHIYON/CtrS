"""Reuse the 65 recipe for a short fine-tuning: the 58 C2MS trainer with the same five L1 replacements as 65/79 (original-pair L1 +
1.0 x coarse-pair L1 + 0.46 x spectral loss; AMP; selection by original-scale validation MSE including epoch 0; EMA recorded only).
The schedule comes from config.json (Adam 5e-5 -> 1e-6 cosine over 10 epochs, 40 for K2L).

Two more replacements, each matching exactly once:
1. make_model loads the factorized initial weights made by `run.py factorize` (runtime/init/<sensor>_<arm>_s<seed>.pt, SHA-256 sealed
   in the runtime config) instead of a random initialization;
2. the preflight initial-state check (4-band models equal to 43) becomes check_initial: the init file hash matches the sealed hash, and
   FT0 reproduces the 65 checkpoint output exactly.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
L1_REPLACEMENTS=(  # identical to 65/79 train.py
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)
INIT_REPLACEMENTS=(
    ("    m=Model(arm,SENSORS[sensor]['bands'],seed).cuda()\n","    m=load_initial(Model(arm,SENSORS[sensor]['bands'],seed),sensor,arm,seed).cuda()\n"),
    ("if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
     "check_initial(cfg,sensor,arm,seed)"),
)
_CFG={}


def init_path(cfg,sensor,arm,seed):return Path(cfg['runtime'])/'init'/f'{sensor}_{arm}_s{seed}.pt'


def load_initial(m,sensor,arm,seed):
    import torch
    cfg=_CFG['cfg'];p=init_path(cfg,sensor,arm,seed);want=cfg.get('init_sha256',{}).get(p.stem)
    if want is None or T.sha(p)!=want:raise RuntimeError('factorized initial weights missing or not sealed: '+str(p))
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True);return m


def check_initial(cfg,sensor,arm,seed):
    """Sealed init hash, and FT0 equal to the 65 checkpoint (same weights, same output)."""
    import torch
    import model as M
    if seed!=cfg['latency_seed']:return  # init files of the stage-2 seeds are checked by hash when their runs start
    p=init_path(cfg,sensor,arm,seed)
    if T.sha(p)!=cfg['init_sha256'][p.stem]:raise AssertionError('init hash '+p.stem)
    if arm=='FT0':
        ref=torch.load(HERE.parents[2]/cfg['init_from'][sensor].replace('<seed>',str(seed)),map_location='cpu',weights_only=False)['model']
        mine=torch.load(p,map_location='cpu',weights_only=False)['model']
        if ref.keys()!=mine.keys() or any(not torch.equal(ref[k],mine[k]) for k in ref):raise AssertionError('FT0 init differs from 65 '+sensor)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS+INIT_REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('58 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_l1s_factorized');mod.__file__=str(SOURCE);mod.load_initial=load_initial;mod.check_initial=check_initial
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):e=cfg['epochs'];return int(e[arm]) if isinstance(e,dict) else int(e)
if __name__=='__main__':
    import argparse
    from model import ARMS
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layout',default='channels_last');p.add_argument('--tag',default='single');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));_CFG['cfg']=cfg
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
