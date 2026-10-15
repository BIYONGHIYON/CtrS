import sys,time,json,copy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from run import StructureNet,seed,write,sha
from common.model import RestoredPansharpeningNet
OUT=Path('C:/CtrS-p150/runtime');exp=Path(__file__).parent
seed();cal=json.loads((exp/'calibration.json').read_text());report={}
base=RestoredPansharpeningNet(4,4).cuda().eval()
for v in ('R0','W2','W3','F01','F11'):
    seed();m=StructureNet(v);m.configure(cal);m=m.cuda();record={'parameters':sum(p.numel() for p in m.parameters()),'macs':{}}
    for size in (64,256,512):
        x=[torch.rand(1,c,s,s,device='cuda') for c,s in [(1,size),(4,size),(4,size//4)]]
        count=[0]
        def hook(mod,inp,out):count[0]+=out.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
        handles=[mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod,torch.nn.Conv2d)]
        m.eval()
        with torch.no_grad():
            y=m(*x);assert torch.isfinite(y).all()
            if v in ('R0','W2','W3','F01'):assert torch.equal(y,base(*x)),(v,size,'initial identity')
        for h in handles:h.remove()
        record['macs'][str(size)]=count[0];del y,x
    if v in ('W2','W3'):
        # At zero head initialization upstream branch gradients emerge after its first update.
        m.train();opt=torch.optim.Adam(m.parameters(),lr=1e-4)
        x=[torch.rand(2,c,s,s,device='cuda') for c,s in [(1,64),(4,64),(4,16)]]
        for step in range(3):
            opt.zero_grad();m(*x).square().mean().backward()
            if step==2:assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.net.backbone.branch.parameters())
            opt.step()
        del x,opt
    del m;torch.cuda.empty_cache();report[v]=record
x=torch.rand(1,64,64,64,device='cuda');assert torch.equal(torch.nn.PixelShuffle(2)(torch.nn.PixelUnshuffle(2)(x)),x)
assert abs(report['W2']['macs']['64']/report['W3']['macs']['64']-1)<.1
write(OUT/'preflight.json',report);print(json.dumps(report),flush=True)
