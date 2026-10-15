"""No-training equation, gradient, scale and serialization checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, torch
from pathlib import Path
from model import PaperSSAMRN, BandAttention


def checks(device):
    torch.set_num_threads(1); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    b = BandAttention(4, 6).double().to(device)
    p, f, m = [torch.randn(2, c, 8, 12, dtype=torch.float64, device=device) for c in (1,4,1)]
    actual = b(p, f, m)
    pp = b.pan2(torch.relu(b.pan1(p))).flatten(2)
    qq = b.spectral(torch.cat((f, m), 1)).flatten(2)
    reference = []
    for n in range(2):
        r = torch.stack([torch.stack([(pp[n,i]*qq[n,j]).sum() for j in range(6)]) for i in range(6)])
        w = r.softmax(-1)
        reference.append(torch.stack([sum(w[i,j]*pp[n,j] for j in range(6)) for i in range(6)]))
    reference = torch.stack(reference).reshape_as(actual)
    err = float((actual-reference).abs().max())
    assert err < 1e-10
    model = PaperSSAMRN().to(device)
    model.train()
    out = model(torch.rand(2,1,64,64,device=device), torch.rand(2,4,64,64,device=device), torch.rand(2,4,16,16,device=device))
    out.square().mean().backward()
    grads = {n: float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
    assert all(v is not None and torch.isfinite(torch.tensor(v)) for v in grads.values())
    assert all(v > 0 for n,v in grads.items() if '.bands.' in n and 'weight' in n)
    model.eval(); shapes=[]
    with torch.inference_mode():
        for h,w in ((64,64),(256,256),(512,512),(32,48)):
            inp=(torch.rand(1,1,h,w,device=device),torch.rand(1,4,h,w,device=device),torch.rand(1,4,h//4,w//4,device=device))
            y=model(*inp); assert y.shape==(1,4,h,w) and torch.isfinite(y).all()
            # H5 LMS must not leak into a paper-bicubic forward.
            alt=model(inp[0], torch.zeros_like(inp[1]), inp[2]); assert torch.equal(y,alt)
            shapes.append([h,w])
        clone=PaperSSAMRN().to(device);clone.load_state_dict(model.state_dict(),strict=True);clone.eval()
        assert torch.equal(y,clone(*inp))
    return {'passed':True,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'equation_float64_max_abs_error':err,'finite_backward_all_parameters':True,'nonzero_attention_weight_gradients':True,'shapes':shapes,'strict_load_forward_equal':True,'supplied_lms_ignored':True}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--device',default='cuda');a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    result=checks(args.device);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
