"""Reuse the 54 C2MS trainer (as 77) with the teacher term added; everything else is unchanged (Adam 1e-4 -> 1e-6 cosine, 150 epochs,
FP16 AMP, original + 1.0 x 2x-coarser pairs, selection by original-scale validation MSE against GT).

Replacements (each must match exactly once in the 54 source):
1. the teacher caches (runtime/cache/<sensor>/<coarse_>{train,val}_teacher.npy, built by run.py prepare) are opened next to the inputs;
2. loss per scale = (sum (p - gt)^2 + lambda * sum (p - teacher)^2) / elements, lambda = config teacher_weight; the logged train_mse and
   coarse_train_mse stay the GT terms, so they are comparable with the controls;
3. each epoch (and epoch 0) also records the validation MSE against the teacher (record only, not used for selection);
4. preflight initial-state check: the student's initial state hash must equal the recorded initialization of its same-structure
   control (77 E3 / 75 W64D8, seed 42); other seeds are compared with a fresh build of the control structure.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'54_multi_sensor_scale_mixed_training/train.py'
REPLACEMENTS=(
    ("n=len(tr['pan']);steps=math.ceil(n/cfg['batch']);cw=weight(cfg,arm)",
     "tt=teacher(cfg,sensor,'train');ttc=teacher(cfg,sensor,'train',True);vt=teacher(cfg,sensor,'val');vtc=teacher(cfg,sensor,'val',True);lam=float(cfg['teacher_weight'])\n"
     "    if not (len(tt)==len(ttc)==len(tr['pan']) and len(vt)==len(vtc)==len(va['pan'])):raise AssertionError('teacher cache length differs from the input cache')\n"
     "    n=len(tr['pan']);steps=math.ceil(n/cfg['batch']);cw=weight(cfg,arm)"),
    ("history=[dict(epoch=0,train_mse=None,**best,**{'coarse_'+k:v for k,v in vc0.items()})]",
     "history=[dict(epoch=0,train_mse=None,**best,**{'coarse_'+k:v for k,v in vc0.items()},val_teacher_mse=teacher_validation(m,va,vt,cfg),coarse_val_teacher_mse=teacher_validation(m,vac,vtc,cfg))]"),
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()",
     "sq=(p.float()-b[3]).square().sum();loss=(sq+lam*(p.float()-tbatch(tt,idx,cfg['layout'])).square().sum())/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum();loss=loss+cw*sqc/bc[3].numel()",
     "sqc=(pc.float()-bc[3]).square().sum();loss=loss+cw*(sqc+lam*(pc.float()-tbatch(ttc,idx,cfg['layout'])).square().sum())/bc[3].numel()"),
    ("v=validation(m,va,cfg);vc=validation(m,vac,cfg);epoch+=1",
     "v=validation(m,va,cfg);vc=validation(m,vac,cfg);tv=teacher_validation(m,va,vt,cfg);tvc=teacher_validation(m,vac,vtc,cfg);epoch+=1"),
    ("lr=lr,order_sha256=order_hash",
     "lr=lr,val_teacher_mse=tv,coarse_val_teacher_mse=tvc,order_sha256=order_hash"),
    ("if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
     "if SENSORS[sensor]['bands']==4:check_initial(arm,seed,sd)"),
)
# Added to the 54 module namespace (they use its np, torch, Path and batch).
EXTRA='''
def teacher(cfg,sensor,split,coarse=False):return np.load(Path(cfg['runtime'])/'cache'/sensor/f"{'coarse_' if coarse else ''}{split}_teacher.npy",mmap_mode='r')
def tbatch(t,idx,layout):
    x=torch.from_numpy(np.array(t[idx],copy=True,order='C')).pin_memory().to('cuda',non_blocking=True)
    if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last)
    return x
@torch.inference_mode()
def teacher_validation(m,c,t,cfg):
    m.eval();acc=torch.zeros((),device='cuda',dtype=torch.float64);elements=0
    for i in range(0,len(c['pan']),cfg['micro']):
        b=batch(c,slice(i,i+cfg['micro']),cfg['layout']);p=m(*b[:3]).float();tb=tbatch(t,slice(i,i+cfg['micro']),cfg['layout'])
        acc+=(p-tb).square().double().sum();elements+=tb.numel()
    return float(acc)/elements
'''


def check_initial(arm,seed,sd):
    import model as M
    conf=json.loads((HERE/'config.json').read_text(encoding='utf-8'));rec=conf['same_structure_initialization'][arm]
    if seed==rec['seed']:
        want=json.loads((HERE.parents[2]/rec['path']).read_text(encoding='utf-8'))['state_sha256']
    else:
        want=T.state_hash(M.base_model(arm,4,seed).state_dict())
    if T.state_hash(sd)!=want:raise AssertionError('student initial state differs from its same-structure control '+arm)


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in REPLACEMENTS:
        if src.count(old)!=1:raise RuntimeError('54 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms54_train_distill');mod.__file__=str(SOURCE);mod.check_initial=check_initial
    exec(compile(src+EXTRA,str(SOURCE),'exec'),mod.__dict__);return mod


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
