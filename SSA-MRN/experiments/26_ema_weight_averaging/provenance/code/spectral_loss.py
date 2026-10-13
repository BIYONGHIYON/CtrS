"""Only the objective changes. FP32 normalized multispectral vectors, channel axis=1."""
import torch
from torch.nn import functional as F

EPS = 1e-4  # normalized QB vector norm; ~= 0.2047 raw DN norm

def spectral_components(prediction, target):
    """Return summed 1-cosine and GT-only valid count over all N,H,W pixels.

    GT norm <= EPS has no informative direction and is excluded, identically for
    both arms. Prediction norm is NEVER used as a mask. A zero prediction against
    valid GT therefore has penalty 1 and a finite, nonzero gradient. Norm floors
    bound gradients; all arithmetic is FP32. No clipping of image values occurs.
    """
    pred=prediction.float(); gt=target.float()
    valid=torch.linalg.vector_norm(gt,dim=1)>EPS
    cosine=F.cosine_similarity(pred,gt,dim=1,eps=EPS).clamp(-1.,1.)
    terms=(1.-cosine)*valid
    return terms.sum(), valid.sum()

def spectral_mean(prediction,target):
    total,count=spectral_components(prediction,target)
    return total/count.clamp_min(1)

def training_group_backward(model, batch, optimizer, weight, micro):
    """CPU/GPU test helper, same weighting as worker: exact effective-batch means."""
    optimizer.zero_grad(set_to_none=True)
    n=len(batch['gt'])
    valid=(torch.linalg.vector_norm(batch['gt'].float(),dim=1)>EPS).sum().clamp_min(1)
    for start in range(0,n,micro):
        b={k:v[start:start+micro] for k,v in batch.items()}
        pred=model(b['pan'],b['lms'],b['ms'])
        objective=F.mse_loss(pred,b['gt'])*len(b['gt'])/n
        if weight:
            total,_=spectral_components(pred,b['gt'])
            objective=objective+weight*total/valid
        objective.backward()
    optimizer.step()
