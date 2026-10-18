"""Joint per-sample photometric change with loss measured back in original DN/2047 units."""
import torch

def make_plan(n,generator,device):
    # A separate stream; consume identical plans in both arms, independent of model and order RNG.
    u=torch.rand((n,2),device=device,generator=generator,dtype=torch.float32)
    return torch.stack((0.9+0.2*u[:,0],-0.01+0.02*u[:,1]),dim=1)

def forward_training(model,batch,plan,enabled):
    # Caller owns AMP around the network; transforms/inverse always FP32.
    if not enabled:
        return model(batch['pan'],batch['lms'],batch['ms']).float()
    with torch.autocast(device_type=batch['ms'].device.type,enabled=False):
        gain=plan[:,0].view(-1,1,1,1);bias=plan[:,1].view(-1,1,1,1)
        inputs=tuple(batch[k].float()*gain+bias for k in ('pan','lms','ms'))
    prediction=model(*inputs)
    with torch.autocast(device_type=batch['ms'].device.type,enabled=False):
        return (prediction.float()-bias)/gain

def plan_statistics(plan,enabled):
    # One transfer per epoch, never per micro batch. Baseline applies exactly identity.
    if not enabled:return {'gain_mean':1.,'gain_min':1.,'gain_max':1.,'bias_mean':0.,'bias_min':0.,'bias_max':0.,'samples':len(plan)}
    p=plan.detach().cpu()
    return {'gain_mean':float(p[:,0].mean()),'gain_min':float(p[:,0].min()),'gain_max':float(p[:,0].max()),
            'bias_mean':float(p[:,1].mean()),'bias_min':float(p[:,1].min()),'bias_max':float(p[:,1].max()),'samples':len(plan)}
