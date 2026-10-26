"""Reuse the original 54 C2MS trainer unchanged (MSE on the original and 2x coarser pairs) for C2MS_AM.

Only the preflight initial-weight check is adapted: the 54 check compares the whole initial state with 43 C2, while
C2MS_AM has extra modulation weights, so the replacement compares the shared C2MS weights (they must be identical)
and the replacement must match exactly once. Training, AMP, data order, validation and selection are the 54 code.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'54_multi_sensor_scale_mixed_training/train.py'
REPLACEMENTS=(
    ("h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict())","(lambda ref:state_hash({k:sd[k] for k in ref})!=state_hash(ref))(M54.M43.Model(M54.CORE43[arm],seed).state_dict())"),
)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('54 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms54_train_am');mod.__file__=str(SOURCE)
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):return int(cfg['epochs'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=['C2MS_AM']);p.add_argument('--seed',type=int,default=42)
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
