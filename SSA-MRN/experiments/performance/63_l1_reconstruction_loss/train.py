"""Reuse the original 54 C2MS trainer with only its loss changed from MSE to L1 (original and 2x coarser pairs).

The 54 train.py source is loaded and exactly four squared-error expressions (training step, coarse term, benchmark and
preflight) are replaced by absolute errors; every replacement must match exactly once, so the rest of the trainer
(optimizer, AMP, data order, validation, selection by validation MSE, resume) is unchanged. History fields keep the 54
names train_mse / coarse_train_mse but hold mean absolute errors.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'54_multi_sensor_scale_mixed_training/train.py'
L1_REPLACEMENTS=(
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('54 trainer changed; L1 replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms54_train_l1');mod.__file__=str(SOURCE)
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):return int(cfg['epochs'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=['C2MS']);p.add_argument('--seed',type=int,default=42)
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
