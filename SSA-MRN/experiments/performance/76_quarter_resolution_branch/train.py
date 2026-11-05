"""Reuse the 54 C2MS trainer unchanged (MSE on the original and 2x coarser pairs, Adam 1e-4 -> 1e-6 cosine, 150 epochs, AMP,
selection by validation MSE). The only replacement is the preflight initial-state check: the three-branch models do not exist in
experiment 43, so the original backbone and W2 branch must equal their source of the same seed (43 C2 for C2MS_W4, 75 C2MS_W64D8 for
C2MS_W64D8_W4), the original part must equal 43 C0, and the initial output must equal C0 exactly.

The replacement must match exactly once, so the rest of the trainer is unchanged.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'54_multi_sensor_scale_mixed_training/train.py'
REPLACEMENTS=(
    ("if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
     "if SENSORS[sensor]['bands']==4:check_initial(arm,seed,sd)"),
)


def check_initial(arm,seed,sd):
    """Original + W2 equal their same-seed source, the original part equals 43 C0, and the zero last projections make the
    initial output equal C0 bit for bit."""
    import torch
    import model as M
    inner={k.replace('core.backbone.inner.','core.backbone.'):v for k,v in sd.items() if not k.startswith('core.backbone.quarter.')}
    src=(M.M43.Model('C2',seed) if arm=='C2MS_W4' else M.M75.Model('C2MS_W64D8',4,seed)).state_dict()
    if inner.keys()!=src.keys() or T.state_hash(inner)!=T.state_hash(src):raise AssertionError('original + W2 part differs from its source '+arm)
    c0=M.M43.Model('C0',seed);ref=c0.state_dict()
    core={k.replace('core.backbone.original.','core.backbone.'):v for k,v in inner.items() if not k.startswith('core.backbone.branch.')}
    if core.keys()!=ref.keys() or T.state_hash(core)!=T.state_hash(ref):raise AssertionError('original part differs from 43 C0 '+arm)
    m=M.Model(arm,4,seed).eval();c0.eval();g=torch.Generator().manual_seed(seed)
    x=[torch.rand(s,generator=g) for s in [(2,1,64,64),(2,4,64,64),(2,4,16,16)]]
    with torch.inference_mode():
        if not torch.equal(m(*x),c0(*x)):raise AssertionError('initial output differs from 43 C0 '+arm)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('54 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms54_train_quarter');mod.__file__=str(SOURCE);mod.check_initial=check_initial
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):return int(cfg['epochs'])
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
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
