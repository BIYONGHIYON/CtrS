"""Small numerical/gradient verification, never reports research performance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.ndimage import convolve1d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ssamrn.models.pan_variants import make_model, HighFrequency
from ssamrn.models.interp23 import interp23tap


def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args()
    device='cuda' if a.cuda else 'cpu'
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
    x=torch.randn(1,2,8,8)
    half=np.array([.5,.305334091185,0,-.072698593239,0,.021809577942,0,-.005192756653,0,.000807762146,0,-.000060081482])
    taps=2*np.r_[half[:0:-1],half]
    for ratio in (2,4):
        ref=x.numpy().copy()
        for stage in range(int(np.log2(ratio))):
            n,c,h,w=ref.shape;up=np.zeros((n,c,h*2,w*2),dtype=np.float32);offset=1 if stage==0 else 0
            up[...,offset::2,offset::2]=ref
            ref=convolve1d(convolve1d(up,taps,axis=-2,mode='wrap'),taps,axis=-1,mode='wrap')
        result=interp23tap(x.to(device),ratio).cpu().numpy()
        np.testing.assert_allclose(result,ref,rtol=1e-5,atol=2e-6)
    pan=torch.rand(2,1,64,64,device=device);lms=torch.rand(2,4,64,64,device=device);ms=torch.rand(2,4,16,16,device=device)
    base=None;core=None
    for variant in ('baseline','interp23','lr_correction','high_frequency'):
        torch.manual_seed(42)
        m=make_model(dict(sensor='QB',k=6,variant=variant,lr_gain=.1)).to(device)
        if core is None:core={k:v.clone() for k,v in m.state_dict().items()}
        else:
            for key,value in core.items():assert torch.equal(value,m.state_dict()[key]),key
        output=m(pan,lms,ms)
        assert output.shape==lms.shape and torch.isfinite(output).all()
        if variant=='baseline':base=output.detach()
        if variant=='high_frequency':assert torch.equal(base,output.detach()),'zero-initialized residual must reproduce baseline'
        if variant=='lr_correction':
            observed=m.observation(lms)
            correction=m.correction(lms,lms,observed)
            assert correction.abs().max()<1e-6,'zero observation error must yield zero correction'
        output.square().mean().backward()
        gradients=[param.grad for param in m.parameters() if param.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        print('PASS',variant,'shape, shared initialization, finite deterministic gradients',flush=True)
        del m,output
    hp=HighFrequency.highpass(torch.ones_like(pan))
    assert hp.abs().max()<1e-7
    print('PASS 23tap SciPy reference x2/x4, constant-PAN highpass, LR zero-error identity; verification only')


if __name__=='__main__':main()
