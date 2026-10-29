"""Reuse the 58 C2MS trainer (54 trainer + 57 spectral loss + EMA record) with the 69 reconstruction losses.

The 58 train.py source is loaded and patched by exact replacements (each must match the expected number of times):
- reconstruction: the squared errors become recon(cfg, arm, sensor, e, phase), an elementwise tensor whose mean is the arm's
  loss for e = prediction - GT (FP32, outside autocast):
    L1_OBS (C2MS_L1SO): |e| + lambda_O x mean (D_p(e))^2, D_p = MS MTF blur and x4 decimation at the patch's phase p
                        (equal to (D_p(y) - MS)^2 inside the margin because the given MS is D_p(GT), prepare.py --check);
    SPLIT (C2MS_SPLIT):  k_S x (L(e))^2 + |e - L(e)|, L = the same MS MTF blur without decimation (60/68 low band);
  the 2x coarser pairs use the same forms with phase 2 (their MS is made by the 54 Wald operator at phase 2);
- the per-patch phase table of the training split (run.py prepare, from the float64 H5) is loaded once per run.
The spectral weight comes from config spectral_weight (0.46 for both arms, as 65). Everything else (optimizer, AMP, data
order, coarse pairs, validation, selection by validation MSE, EMA record) is the 58 trainer. The history fields
train_mse/coarse_train_mse hold the arm's reconstruction loss.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,traceback,types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
from prepare import Wald,SENSORS,_blur,decimate,find_phases,data_files
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
REPLACEMENTS=(
    # training step (original pair with the patch phases, coarse pair at phase 2)
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=recon(cfg,arm,sensor,p.float()-b[3],PH[idx]).sum();loss=sq/b[3].numel()",1),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=recon(cfg,arm,sensor,pc.float()-bc[3]).sum()",1),
    # throughput benchmark and preflight probe (a fixed phase is enough there)
    ("loss=(p.float()-b[3]).square().mean()","loss=recon(cfg,arm,sensor,p.float()-b[3],np.full(len(idx),10)).mean()",1),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*recon(cfg,arm,sensor,pc.float()-bc[3]).mean()",1),
    # preflight probe: two real training patches (and their coarse pairs, GT and phases) instead of uniform noise, because the
    # low-band squared term (weight 500 / 431.5) on noise-sized errors overflows the FP16 gradients of the first AMP step
    ("inputs=[torch.rand(s,device='cuda') for s in [(2,1,64,64),(2,nb,64,64),(2,nb,16,16)]];coarse_in=[torch.rand(s,device='cuda') for s in [(2,1,32,32),(2,nb,32,32),(2,nb,8,8)]]",
     "inputs=[x[:2].contiguous() for x in b[:3]];coarse_in=[x[:2].contiguous() for x in bc[:3]];G=b[3][:2];GC=bc[3][:2];PHP=phase_table(cfg,sensor,len(tr['pan']))[idx[:2]]",1),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()+spectral_weight(cfg,arm)*spectral_loss(p.float(),inputs[1])",
     "loss=recon(cfg,arm,sensor,p.float()-G,PHP).mean()+cw*recon(cfg,arm,sensor,pc.float()-GC).mean()+spectral_weight(cfg,arm)*spectral_loss(p.float(),G)",1),
    # preflight probe: start the AMP scale at 64 instead of 1024. At initialization the low-band squared term gives a loss of
    # about 18 and parameter gradients up to about 84 on real patches, so 1024 overflows FP16 (checked on the server GPU, finite
    # at 64). Training keeps the 58 start scale 1024; GradScaler skips the overflowing first steps and halves the scale (recorded).
    ("scaler=torch.amp.GradScaler('cuda',init_scale=1024.)\n            def step():","scaler=torch.amp.GradScaler('cuda',init_scale=64.)\n            def step():",1),
    # phase table of the training split, loaded with the caches
    ("tr=cache(cfg,sensor,'train');va=cache(cfg,sensor,'val');","tr=cache(cfg,sensor,'train');PH=phase_table(cfg,sensor,len(tr['pan']));va=cache(cfg,sensor,'val');",1),
)
_KERNELS={}


def kernel(sensor,device):
    """Per-band MS MTF kernel of the 54 Wald operator (genMTF ratio 4, 41 taps), float32 on the device."""
    key=(sensor,str(device))
    if key not in _KERNELS:_KERNELS[key]=Wald(sensor,'cpu').ms_kernel.float().to(device)
    return _KERNELS[key]


def recon(cfg,arm,sensor,e,phase=None):
    """Elementwise reconstruction loss of e = prediction - GT; its mean is the arm's loss (phase None = phase 2)."""
    kind=cfg['reconstruction'][arm]
    if kind=='MSE':return e.square()
    if kind=='L1':return e.abs()
    low=_blur(e,kernel(sensor,e.device))
    if kind=='L1_OBS':return e.abs()+float(cfg['loss_weights']['lambda_O'])*decimate(low,phase).square().mean()
    if kind=='SPLIT':return float(cfg['loss_weights']['k_S'])*low.square()+(e-low).abs()
    raise ValueError(kind)


def phase_table(cfg,sensor,n):
    ph=np.load(Path(cfg['runtime'])/'cache'/sensor/'train_phase.npy')
    if ph.shape!=(n,) or ph.min()<0 or ph.max()>15:raise ValueError('phase table does not match the training cache '+sensor)
    return ph


def loss_check(cfg):
    """CPU checks of the 69 losses and phase tables, written to preflight/loss_checks.json (part of preflight)."""
    import h5py
    torch.manual_seed(7);res={}
    # 1. decimate() at per-sample phases equals explicit slicing; the loss forms equal their formulas.
    x=torch.rand(16,4,64,64,dtype=torch.float64);ph=np.arange(16)
    ref=torch.stack([x[i,:,q//4::4,q%4::4] for i,q in enumerate(ph)])
    if float((decimate(x,ph)-ref).abs().max())!=0:raise AssertionError('per-sample decimation')
    if float((decimate(x)-x[...,2::4,2::4]).abs().max())!=0:raise AssertionError('phase-2 decimation')
    e=torch.randn(4,4,64,64)*.01;p4=np.array([6,9,10,10]);k=kernel('QB','cpu');low=_blur(e,k)
    lam,ks=float(cfg['loss_weights']['lambda_O']),float(cfg['loss_weights']['k_S'])
    forms={'C2MS_L1SO':float(e.abs().mean()+lam*torch.stack([low[i,:,q//4::4,q%4::4] for i,q in enumerate(p4)]).square().mean()),
           'C2MS_SPLIT':float(ks*low.square().mean()+(e-low).abs().mean())}
    for arm,v in forms.items():
        got=float(recon(cfg,arm,'QB',e,p4).mean());got_sum=float(recon(cfg,arm,'QB',e,p4).sum()/e.numel())
        if abs(got-v)>1e-9*max(1,abs(v)) or abs(got_sum-v)>1e-6*max(1,abs(v)):raise AssertionError('loss form '+arm)
        res['form_'+arm]=dict(formula=v,recon_mean=got)
    # 2. Per sensor: the phase table makes D_p(GT) equal the given MS inside the margin (float64 H5, 64 evenly spaced
    #    patches), the table rows follow the cache rows, and the coarse MS equals D(coarse GT) at phase 2.
    for sensor in cfg['sensors']:
        tr={key:np.load(Path(cfg['runtime'])/'cache'/sensor/f'train_{key}.npy',mmap_mode='r') for key in ('gt','ms')}
        n=len(tr['gt']);PH=phase_table(cfg,sensor,n);idx=np.linspace(0,n-1,64).astype(int);s=SENSORS[sensor]['max_value']
        with h5py.File(Path(cfg['data_root'])/data_files(sensor)['train'],'r') as f:
            gt=torch.from_numpy(np.asarray(f['gt'][idx],dtype=np.float64));ms=torch.from_numpy(np.asarray(f['ms'][idx],dtype=np.float64))
        kd=Wald(sensor,'cpu').ms_kernel;worst=0.;fixed=0.
        for j in range(0,64,8):
            b=_blur(gt[j:j+8],kd);m=ms[j:j+8]
            worst=max(worst,float((decimate(b,PH[idx[j:j+8]])-m)[...,5:-5,5:-5].abs().max()))
            fixed=max(fixed,float((decimate(b)-m)[...,5:-5,5:-5].abs().max()))
        if worst>1e-6:raise AssertionError(f'phase table: interior mismatch {worst} DN '+sensor)
        cache_vs_h5=float((torch.from_numpy(np.array(tr['gt'][idx],dtype=np.float64))*s-gt).abs().max())
        if cache_vs_h5>s*1e-6:raise AssertionError('cache rows differ from the H5 rows '+sensor)
        trc={key:np.load(Path(cfg['runtime'])/'cache'/sensor/f'coarse_train_{key}.npy',mmap_mode='r') for key in ('gt','ms')}
        g2=torch.from_numpy(np.array(trc['gt'][idx[:16]],dtype=np.float64));m2=torch.from_numpy(np.array(trc['ms'][idx[:16]],dtype=np.float64))
        coarse=float((decimate(_blur(g2,kd))-m2).abs().max())
        if coarse>1e-5:raise AssertionError('coarse MS is not D(coarse GT) at phase 2 '+sensor)
        res[sensor]=dict(patches=64,interior_max_abs_DN_with_table=worst,interior_max_abs_DN_at_phase2=fixed,cache_vs_h5_max_abs_DN=cache_vs_h5,
                         coarse_max_abs_normalized=coarse,phase_counts={str(q):int(c) for q,c in enumerate(np.bincount(PH,minlength=16)) if c})
    return res


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new,count in REPLACEMENTS:
        if src.count(old)!=count:raise RuntimeError(f'58 trainer changed; replacement found {src.count(old)} times, expected {count}: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_69');mod.__file__=str(SOURCE)
    mod.recon=recon;mod.phase_table=phase_table
    exec(compile(src,str(SOURCE),'exec'),mod.__dict__);return mod


T=_load()
T.HERE=HERE
KEYS,sha,write,lock,now,KST,run_name,ARMS=T.KEYS,T.sha,T.write,T.lock,T.now,T.KST,T.run_name,T.ARMS
def epochs(cfg,arm):return int(cfg['epochs'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['train','bench','preflight','evaluate'],required=True)
    p.add_argument('--sensor');p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layout',default='channels_last');p.add_argument('--tag',default='single');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    try:
        if a.mode=='bench':T.benchmark(cfg,a.sensor,a.arm,a.layout,a.tag)
        elif a.mode=='preflight':
            write(Path(cfg['runtime'])/'preflight/loss_checks.json',dict(time_kst=now(),checks=loss_check(cfg)))
            T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
