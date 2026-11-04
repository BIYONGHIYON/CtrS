"""Reuse the 54 C2MS trainer with three changes: L1 instead of MSE (the five replacements of experiment 63), the learning rate
of each arm from config schedule_params, and a stability check after epoch 3 for arms that declare one.

Every replacement must match exactly once, so the rest of the trainer (optimizer, AMP, data order, validation, selection by
validation MSE, resume) is unchanged. History fields keep the 54 names train_mse / coarse_train_mse but hold mean absolute errors.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'54_multi_sensor_scale_mixed_training/train.py'
REPLACEMENTS=(
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
    ("lr=cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*epoch/(cfg['epochs']-1)))","lr=schedule(cfg,arm,epoch)"),
    ("log('epoch_complete',**row,best_epoch=best_epoch)","log('epoch_complete',**row,best_epoch=best_epoch);stability(cfg,arm,history,out)"),
)


def schedule(cfg,arm,epoch):
    """Learning rate of the zero-based epoch."""
    s=cfg['schedule_params'][arm]
    if s['kind']=='step':return s['start']*s['factor']**(epoch//s['every'])
    if s['kind']=='cosine':return s['end']+.5*(s['start']-s['end'])*(1+math.cos(math.pi*epoch/(cfg['epochs']-1)))
    raise ValueError(s['kind'])


def stability(cfg,arm,history,out):
    """After the epoch named in the config: finite losses, skipped AMP updates within the limit, training loss falling. Stops the run otherwise."""
    rule=cfg['schedule_params'][arm].get('stability')
    if not rule or history[-1]['epoch']!=rule['epoch']:return
    rows={r['epoch']:r for r in history}; last=rows[rule['epoch']]
    losses=[r['train_mse'] for r in history if r['train_mse'] is not None]
    result=dict(epoch=rule['epoch'],finite=all(math.isfinite(x) for x in losses),skipped_ratio=last['skipped']/last['attempted'],
                loss_first=rows[1]['train_mse'],loss_last=last['train_mse'])
    result['passed']=result['finite'] and result['skipped_ratio']<=rule['max_skipped_ratio'] and result['loss_last']<result['loss_first']
    T.write(out/'stability_check.json',dict(time_kst=T.now(),**result))
    if not result['passed']:raise RuntimeError('stability check failed; stopping without changing the learning rate: '+json.dumps(result))


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('54 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms54_train_l1_schedule');mod.__file__=str(SOURCE);mod.schedule=schedule;mod.stability=stability
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name
def epochs(cfg,arm):return int(cfg['epochs'])
if __name__=='__main__':
    import argparse
    from model import ARMS
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate','summary'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layout',default='channels_last');p.add_argument('--tag',default='single');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    try:
        if a.mode=='bench':T.benchmark(cfg,a.sensor,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        elif a.mode=='summary':
            from evaluate import summary
            summary(cfg,a.sensor,a.seed)
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+(run_name(a.sensor,a.arm,a.seed) if a.arm else 'summary'))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
