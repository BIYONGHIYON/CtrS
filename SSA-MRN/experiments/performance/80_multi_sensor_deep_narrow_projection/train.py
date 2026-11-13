"""Reuse the 65/79 recipe unchanged on GF2 and WV3: the 58 C2MS trainer (54 trainer + 57 spectral loss + EMA record) with the
reconstruction loss changed from MSE to L1 by the same five replacements as 65/79 train.py (original-pair L1 + 1.0 x coarse-pair L1 +
0.46 x spectral loss on the original pair; Adam 1e-4 -> 1e-6 cosine, 150 epochs, AMP, selection by original-scale validation MSE).

The preflight initial-state check is replaced as in 79, generalized to any band count (79 checked 4-band models only): every layer outside
the W2 branch must equal 54 C0 of the same seed and band count, the first 8 branch blocks and both 1x1 layers must equal 75 C2MS_W64D8 of the
same seed and band count (same RNG order), and the initial output must equal 54 C0 exactly (the branch's last 1x1 starts at zero). On 4-band
sensors 54 C0 equals 43 C0, so the GF2 check is the 79 check. Each replacement must match exactly once.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
L1_REPLACEMENTS=(  # identical to 65 and 79 train.py
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=(p.float()-b[3]).abs().sum();loss=sq/b[3].numel()"),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=(pc.float()-bc[3]).abs().sum()"),
    ("loss=(p.float()-b[3]).square().mean()","loss=(p.float()-b[3]).abs().mean()"),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*(pc.float()-bc[3]).abs().mean()"),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=(p.float()-inputs[1]).abs().mean()+cw*(pc.float()-coarse_in[1]).abs().mean()"),
)
INITIAL_REPLACEMENT=(
    "if SENSORS[sensor]['bands']==4 and h!=state_hash(M54.M43.Model(M54.CORE43[arm],seed).state_dict()):raise AssertionError('4-band initial state differs from 43 '+arm)",
    "check_initial(arm,seed,sd,SENSORS[sensor]['bands'])")


def check_initial(arm,seed,sd,bands):
    import torch
    import model as M
    c0=M.M54.Model('C0',bands,seed);ref=c0.state_dict();w8=M.M75.Model('C2MS_W64D8',bands,seed).state_dict()
    outside={k:v for k,v in sd.items() if not k.startswith('core.backbone.branch.')}
    ref_outside={k.replace('core.backbone.','core.backbone.original.') if k.startswith('core.backbone.') else k:v for k,v in ref.items()}
    if outside.keys()!=ref_outside.keys() or T.state_hash(outside)!=T.state_hash(ref_outside):raise AssertionError(f'layers outside the W2 branch differ from 54 C0 {arm} {bands} bands')
    # 75 W64D8 branch: [0] PixelUnshuffle, [1] 1x1 in, [2..9] blocks, [10] 1x1 out, [11] PixelShuffle; ours has depth blocks before the 1x1 out
    depth=M.SHAPES[arm][1];shared={}
    for k,v in w8.items():
        if not k.startswith('core.backbone.branch.'):continue
        i=int(k.split('.')[3]);mine=k if i<10 else k.replace(f'branch.{i}.',f'branch.{i-8+depth}.',1)
        shared[mine]=v
    if any(T.state_hash({k:sd[k]})!=T.state_hash({k:v}) for k,v in shared.items()):raise AssertionError(f'W2 branch differs from 75 C2MS_W64D8 {arm} {bands} bands')
    m=M.Model(arm,bands,seed).eval();c0.eval();g=torch.Generator().manual_seed(seed)
    x=[torch.rand(s,generator=g) for s in [(2,1,64,64),(2,bands,64,64),(2,bands,16,16)]]
    with torch.inference_mode():
        if not torch.equal(m(*x),c0(*x)):raise AssertionError(f'initial output differs from 54 C0 {arm} {bands} bands')


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new in L1_REPLACEMENTS+(INITIAL_REPLACEMENT,):
        if src.count(old)!=1:raise RuntimeError('58 trainer changed; replacement not unique: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_l1s_deep_narrow_multi');mod.__file__=str(SOURCE);mod.check_initial=check_initial
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
