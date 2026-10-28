"""Reuse the 58 C2MS trainer (54 trainer + 57 spectral loss + EMA record) with a per-arm reconstruction loss and a PAN input augmentation.

The 58 train.py source is loaded and patched by exact replacements (each must match the expected number of times):
- reconstruction: the squared errors become recon(cfg, arm, e), i.e. e.square() for C2MS_* (MSE, as C2MS) and e.abs()
  for L1S_* (L1, as 65); the spectral weight comes from config spectral_weight (0 for C2MS_*, 0.46 for L1S_*);
- augmentation: the PAN of every training batch, on the original and on the 2x coarser pair, goes through
  augment_pan(); GT, MS and LMS are unchanged, validation and test inputs are never augmented;
- the augmentation draws come from their own CPU generator (seed + 200000), saved in latest.pt for an exact resume.
Everything else (optimizer, AMP, data order, coarse pairs, validation, selection by validation MSE) is the 58 trainer.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,sys,traceback,types
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))
SOURCE=HERE.parent/'58_spectral_loss_multi_sensor/train.py'
REPLACEMENTS=(
    # reconstruction loss per arm (training step, coarse term, benchmark, preflight)
    ("sq=(p.float()-b[3]).square().sum();loss=sq/b[3].numel()","sq=recon(cfg,arm,p.float()-b[3]).sum();loss=sq/b[3].numel()",1),
    ("sqc=(pc.float()-bc[3]).square().sum()","sqc=recon(cfg,arm,pc.float()-bc[3]).sum()",1),
    ("loss=(p.float()-b[3]).square().mean()","loss=recon(cfg,arm,p.float()-b[3]).mean()",1),
    ("loss=loss+cw*(pc.float()-bc[3]).square().mean()","loss=loss+cw*recon(cfg,arm,pc.float()-bc[3]).mean()",1),
    ("loss=(p.float()-inputs[1]).square().mean()+cw*(pc.float()-coarse_in[1]).square().mean()","loss=recon(cfg,arm,p.float()-inputs[1]).mean()+cw*recon(cfg,arm,pc.float()-coarse_in[1]).mean()",1),
    # PAN augmentation in the training loop (original and coarse pair) and in the throughput benchmark
    ("idx=order[bi*cfg['batch']:(bi+1)*cfg['batch']].numpy();b=batch(tr,idx,cfg['layout']);","idx=order[bi*cfg['batch']:(bi+1)*cfg['batch']].numpy();b=batch(tr,idx,cfg['layout']);b[0]=augment_pan(b[0],cfg,arm,agen);",1),
    ("bc=batch(trc,idx,cfg['layout'])\n","bc=batch(trc,idx,cfg['layout']);bc[0]=augment_pan(bc[0],cfg,arm,agen)\n",1),
    ("b=batch(c,idx,layout);opt.zero_grad(set_to_none=True)","b=batch(c,idx,layout);b[0]=augment_pan(b[0],cfg,arm,gen);opt.zero_grad(set_to_none=True)",1),
    ("bc=batch(cc,idx,layout)\n","bc=batch(cc,idx,layout);bc[0]=augment_pan(bc[0],cfg,arm,gen)\n",1),
    # augmentation generator: created with the data-order generator, restored on resume, saved in latest.pt
    ("gen=torch.Generator().manual_seed(seed+100000);","gen=torch.Generator().manual_seed(seed+100000);agen=torch.Generator().manual_seed(seed+200000);",1),
    ("gen.set_state(ck['generator'])","gen.set_state(ck['generator']);agen.set_state(ck['augment_generator'])",1),
    ("generator=gen.get_state(),","generator=gen.get_state(),augment_generator=agen.get_state(),",1),
)


def recon(cfg,arm,e):
    kind=cfg['reconstruction'][arm]
    if kind=='MSE':return e.square()
    if kind=='L1':return e.abs()
    raise ValueError(kind)


def blur_pan(pan,sigma,radius):
    """Per-sample separable Gaussian blur of (N,1,H,W) with replicate borders; sigma 0 is the identity (unit DC gain)."""
    n=pan.shape[0];t=torch.arange(-radius,radius+1,device=pan.device,dtype=torch.float32)
    s=sigma.to(pan.device,torch.float32).clamp_min(1e-6)[:,None]
    k=torch.exp(-t[None]**2/(2*s**2));k=k/k.sum(1,keepdim=True)
    x=F.pad(pan.float().reshape(1,n,*pan.shape[-2:]),(radius,radius,radius,radius),mode='replicate')
    x=F.conv2d(x,k.view(n,1,1,-1),groups=n);x=F.conv2d(x,k.view(n,1,-1,1),groups=n)
    return x.reshape(n,1,*pan.shape[-2:])


def augment_pan(pan,cfg,arm,g):
    """PAN brightness (_G) and/or blur (_GB, _B) per sample, each applied with probability p; other inputs untouched.
    Four uniforms per sample are always drawn (gain on/value, blur on/sigma) so that the same samples get the same gain
    in _G and _GB and the same blur in _B and _GB."""
    a=cfg['augment'][arm];aug=cfg['augmentation'];n=pan.shape[0];p=float(aug['probability'])
    u=torch.rand(n,4,generator=g);out=pan.float()
    if a['blur']:
        sigma=u[:,3]*float(aug['blur_sigma_max'])*(u[:,2]<p)
        out=blur_pan(out,sigma,int(math.ceil(3*float(aug['blur_sigma_max']))))
    if a['gain']:
        log_gain=(2*u[:,1]-1)*float(aug['gain_log_half_range'])*(u[:,0]<p)
        out=out*torch.exp(log_gain).to(out.device).view(n,1,1,1)
    if cfg.get('layout')=='channels_last':out=out.contiguous(memory_format=torch.channels_last)
    return out


def augment_check(cfg):
    """CPU checks of the augmentation, written to preflight/augment_checks.json (part of preflight)."""
    g=lambda: torch.Generator().manual_seed(123);torch.manual_seed(5);pan=torch.rand(4096,1,32,32)
    c={**cfg,'layout':'contiguous'};aug=cfg['augmentation'];res={}
    out={arm:augment_pan(pan,c,arm,g()) for arm in cfg['augment']}
    for arm,y in out.items():
        a=cfg['augment'][arm];ratio=(y/pan).flatten(1)
        if not a['blur']:
            if (ratio.max(1).values-ratio.min(1).values).abs().max()>1e-5:raise AssertionError('gain-only arm changed the PAN spatially '+arm)
            lo,hi=float(ratio[:,0].min()),float(ratio[:,0].max())
            if lo<math.exp(-aug['gain_log_half_range'])-1e-6 or hi>math.exp(aug['gain_log_half_range'])+1e-6:raise AssertionError('gain outside range '+arm)
        changed=float(((y-pan).abs().flatten(1).max(1).values>1e-6).float().mean())
        res[arm]=dict(fraction_changed=changed,mean_ratio=float(y.mean()/pan.mean()))
        q=aug['probability'];expected=q if a['gain']!=a['blur'] else 1-(1-q)**2
        # a blur with sigma below about 0.25 px changes a float32 PAN by less than 1e-6, so blur arms may count up to ~10% fewer changed samples
        low=expected-0.03-(q*0.25/aug['blur_sigma_max'] if a['blur'] else 0)
        if not low<=changed<=expected+0.03:raise AssertionError('augmentation probability '+arm)
    for loss in ('C2MS','L1S'):
        gb,b_,g_=out[loss+'_GB'],out[loss+'_B'],out[loss+'_G']
        if float((gb-b_*(g_/pan).flatten(1)[:,:1].view(-1,1,1,1)).abs().max())>1e-5:raise AssertionError('GB is not B followed by the G gain')
    if float((augment_pan(pan,c,'C2MS_B',g())-out['C2MS_B']).abs().max())!=0:raise AssertionError('augmentation not reproducible from the generator state')
    if float((blur_pan(pan[:3],torch.zeros(3),4)-pan[:3]).abs().max())>1e-6:raise AssertionError('sigma 0 is not the identity')
    flat=torch.full((2,1,32,32),.37);
    if float((blur_pan(flat,torch.tensor([.6,1.2]),4)-flat).abs().max())>1e-6:raise AssertionError('blur DC gain is not 1')
    res['blur_mtf_at_ms_nyquist']={str(s):math.exp(-2*math.pi**2*s**2/64) for s in (0.4,0.8,1.2)}
    return res


def _load():
    src=SOURCE.read_text(encoding='utf-8')
    for old,new,count in REPLACEMENTS:
        if src.count(old)!=count:raise RuntimeError(f'58 trainer changed; replacement found {src.count(old)} times, expected {count}: '+old)
        src=src.replace(old,new)
    mod=types.ModuleType('c2ms58_train_67');mod.__file__=str(SOURCE)
    mod.recon=recon;mod.augment_pan=augment_pan
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
            write(Path(cfg['runtime'])/'preflight/augment_checks.json',dict(time_kst=now(),checks=augment_check(cfg)))
            T.preflight(cfg)
        elif a.mode=='evaluate':
            from evaluate import evaluate
            evaluate(cfg,a.sensor,a.arm,a.seed)
        else:T.train(cfg,a.sensor,a.arm,a.seed,a.resume)
    except BaseException:
        dest=Path(cfg['runtime'])/('preflight' if a.mode in ('bench','preflight') else 'runs/'+run_name(a.sensor,a.arm,a.seed))
        write(dest/f'failure_{a.mode}.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
