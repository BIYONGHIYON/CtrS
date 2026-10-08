"""Verify new interpolation scopes, sensor bands and validation metric definitions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ssamrn.models.pan_variants import make_model
from ssamrn.models.interp23 import Interp23
from train_a6000_followup import validation_metrics
from train import evaluate


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for sensor,channels in [('QB',4),('GF2',4),('WV3',8)]:
        reference=None
        for variant in ['baseline','interp23','interp23_input','interp23_output']:
            torch.manual_seed(42)
            model=make_model(dict(sensor=sensor,k=6,variant=variant,lr_gain=.1)).cuda()
            state=model.state_dict()
            if reference is None:reference={k:v.clone() for k,v in state.items()}
            assert all(torch.equal(v,state[k]) for k,v in reference.items())
            scopes=[isinstance(getattr(model,k),Interp23) for k in ['upsample1','upsample100','upsample101','upsample102']]
            expected={'baseline':[False]*4,'interp23':[True]*4,'interp23_input':[True,True,False,False],'interp23_output':[False,False,True,True]}[variant]
            assert scopes==expected
            tensors=[torch.rand(1,1,64,64,device='cuda'),torch.rand(1,channels,64,64,device='cuda'),torch.rand(1,channels,16,16,device='cuda')]
            out=model(*tensors);assert out.shape==(1,channels,64,64) and torch.isfinite(out).all()
            out.square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            del model,out
        print('PASS',sensor,channels,'bands, shared initialization, four interpolation scopes and gradients',flush=True)
    class Dummy(nn.Module):
        def forward(self,pan,lms,ms):return lms
    gt=torch.rand(2,4,16,16)*.8+.1;pred=gt+torch.randn_like(gt)*.02
    loader=[dict(gt=gt,lms=pred,pan=torch.zeros(2,1,16,16),ms=torch.zeros(2,4,4,4))]
    mse,psnr,sam=validation_metrics(Dummy().cuda(),loader)
    assert abs(mse-evaluate(Dummy().cuda(),loader,'cuda'))<1e-8
    a=gt.numpy().astype('float64');b=pred.numpy().astype('float64')
    expected_psnr=(-10*np.log10(((a-b)**2).mean((-2,-1)))).mean()
    expected_sam=np.degrees(np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1,1))).mean()
    assert abs(psnr-expected_psnr)<1e-4 and abs(sam-expected_sam)<1e-3
    print('PASS validation MSE parity, measured band-mean peak1 PSNR and per-image SAM reference')


if __name__=='__main__':main()
