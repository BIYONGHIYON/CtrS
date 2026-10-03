"""Numerical checks only: no data loader, optimizer or training run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from ssamrn.losses import reconstruction_loss
from ssamrn.models.rgb_grouped import build_rgb_hsi_model

def main():
    gt = torch.rand(2,204,4,4) + .1
    mask = torch.ones(2,4,4,dtype=torch.bool); mask[:,:,0] = False
    loss, mse, spec = reconstruction_loss(gt * 2,gt,mask,.01)
    assert spec.abs() < 1e-6 and mse > 0
    pred=gt.clone();pred[:,:,0,0]=100
    _,mse,spec=reconstruction_loss(pred,gt,mask,.01)
    assert mse == 0 and spec.abs() < 1e-6
    for pred, target in [(torch.zeros_like(gt),gt),(torch.rand_like(gt),torch.zeros_like(gt)),(gt,gt)]:
        pred=pred.clone().requires_grad_()
        loss,_,_=reconstruction_loss(pred,target,mask,.01)
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    pred=torch.rand_like(gt).requires_grad_()
    loss,_,_=reconstruction_loss(pred,gt,torch.zeros_like(mask),.01)
    loss.backward();assert loss == 0 and torch.isfinite(pred.grad).all()
    model=build_rgb_hsi_model(dict(model_type='rgb_triple_grouped17_23tap',latent_channels=17,ssai_dimension=4,upsampler='23tap'))
    model.eval()
    with torch.no_grad():
        result=model(torch.rand(1,3,32,32),torch.rand(1,204,8,8))
    assert result.shape == (1,204,32,32) and torch.isfinite(result).all()
    assert model.encoder.groups==17 and model.encoder.weight.shape[1]==12
    print('PASS: masked spectral loss, finite gradients, 12 bands/group, 17-feature x4 forward; parameters=',sum(p.numel() for p in model.parameters()))

if __name__=='__main__': main()
