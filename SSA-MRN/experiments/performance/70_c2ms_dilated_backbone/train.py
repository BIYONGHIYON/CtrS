"""Reuse the exact experiment-58 trainer with MSE and independently seeded scale-scoped PAN blur."""
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
    ("idx=order[bi*cfg['batch']:(bi+1)*cfg['batch']].numpy();b=batch(tr,idx,cfg['layout']);","idx=order[bi*cfg['batch']:(bi+1)*cfg['batch']].numpy();b=batch(tr,idx,cfg['layout']);b[0]=augment_pan(b[0],cfg,arm,agen,'original');",1),
    ("bc=batch(trc,idx,cfg['layout'])\n","bc=batch(trc,idx,cfg['layout']);bc[0]=augment_pan(bc[0],cfg,arm,agen,'coarse')\n",1),
    ("b=batch(c,idx,layout);opt.zero_grad(set_to_none=True)","b=batch(c,idx,layout);b[0]=augment_pan(b[0],cfg,arm,gen,'original');opt.zero_grad(set_to_none=True)",1),
    ("bc=batch(cc,idx,layout)\n","bc=batch(cc,idx,layout);bc[0]=augment_pan(bc[0],cfg,arm,gen,'coarse')\n",1),
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


def augment_pan(pan,cfg,arm,g,scale="original"):
    n=len(pan);u=torch.rand(n,4,generator=g)
    probability=cfg['blur_probability'][arm][scale]
    if probability==0:return pan
    sigma=u[:,3]*cfg['augmentation']['blur_sigma_max']*(u[:,2]<probability)
    out=blur_pan(pan,sigma,4)
    return out.contiguous(memory_format=torch.channels_last) if cfg['layout']=='channels_last' else out


def augment_check(cfg):
    from model import Model
    torch.manual_seed(7);pan=torch.rand(256,1,32,32);result={}
    for arm in cfg['arms']:
        for scale in ('original','coarse'):
            gen=lambda:torch.Generator().manual_seed(123)
            a=augment_pan(pan,cfg,arm,gen(),scale);b=augment_pan(pan,cfg,arm,gen(),scale)
            assert torch.equal(a,b)
            if cfg['blur_probability'][arm][scale]==0:assert torch.equal(a,pan)
            result[arm+'_'+scale]=float((a-pan).abs().max())
        for bands in (4,8):
            m=Model(arm,bands,42);ref=Model('C2MS',bands,42)
            assert all(torch.equal(v,ref.state_dict()[k]) for k,v in m.state_dict().items())
            for i in (1,3):
                conv=m.core.backbone.original[i].conv21
                expected=2 if arm=='C2MS_D' else 1
                assert conv.dilation==(expected,expected) and conv.padding==(expected,expected)
            clone=Model(arm,bands,42);clone.load_state_dict(m.state_dict(),strict=True)
    flat=torch.full((2,1,32,32),.37)
    assert (blur_pan(flat,torch.tensor([.6,1.2]),4)-flat).abs().max()<1e-6
    return result


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
