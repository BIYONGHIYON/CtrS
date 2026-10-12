"""Meaningful model checks: controls, paired initialization, gradients and recovery."""
import sys,hashlib,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
DualStreamNet = __import__('importlib').import_module('experiments.19_dual_stream_reconstruction.model').DualStreamNet; VARIANTS = __import__('importlib').import_module('experiments.19_dual_stream_reconstruction.model').VARIANTS
from common.model import RestoredPansharpeningNet

def main():
 torch.set_num_threads(1);torch.manual_seed(123)
 inp=[torch.rand(2,c,h,w) for c,h,w in [(1,32,48),(4,32,48),(4,8,12)]];gt=torch.rand(2,4,32,48)
 records={};hashes={};outputs={}
 for v in VARIANTS:
  torch.manual_seed(42);m=DualStreamNet(v);m.train();p=m(*inp);assert p.shape==gt.shape and torch.isfinite(p).all()
  loss=(p-gt).square().mean();loss.backward()
  assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
  if v in ('C2','C3','C4','C5'):
   assert all(x.grad is not None and x.grad.abs().sum()>0 for x in m.parameters()),(v,'dead parameter')
  m.eval()
  with torch.no_grad():
   y=m(*inp);no_pan=m(torch.zeros_like(inp[0]),*inp[1:]);no_ms=m(*inp[:2],torch.zeros_like(inp[2]))
  assert not torch.equal(y,no_pan) and not torch.equal(y,no_ms)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.pt';torch.save(m.state_dict(),path);n=DualStreamNet(v).eval();n.load_state_dict(torch.load(path,weights_only=True),strict=True)
   with torch.no_grad():assert torch.equal(n(*inp),y)
  digest=hashlib.sha256()
  torch.manual_seed(42);init=DualStreamNet(v)
  for name,value in sorted(init.core.state_dict().items()):digest.update(name.encode());digest.update(value.numpy().tobytes())
  hashes[v]=digest.hexdigest();outputs[v]=y;records[v]=dict(parameters=sum(x.numel() for x in m.parameters()),gradients_finite=True,checkpoint_roundtrip_exact=True,pan_and_ms_used=True,rectangular_shape=list(y.shape))
  if v=='C0':
   reference=RestoredPansharpeningNet(4,4).eval();reference.load_state_dict(m.core.state_dict())
   with torch.no_grad():assert torch.equal(reference(*inp),y)
  if v=='C1':
   with torch.no_grad():
    current=inp[1]
    for _ in range(3):current=current+m.alpha*(m.core(inp[0],current+m.error_encoder(torch.zeros_like(current)),inp[2])-current)
    assert torch.equal(current,y)
 assert hashes['C0']==hashes['C1']
 assert len(set(hashes[x] for x in VARIANTS[2:]))==1
 assert not torch.equal(outputs['C2'],outputs['C3']) and not torch.equal(outputs['C2'],outputs['C4']) and not torch.equal(outputs['C3'],outputs['C5'])
 # A full Adam/RNG checkpoint must continue the same update sequence.
 torch.manual_seed(42);m=DualStreamNet('C5');opt=torch.optim.Adam(m.parameters(),lr=1e-4)
 def step(model,optimizer):
  optimizer.zero_grad();((model(*inp)-gt).square().mean()).backward();optimizer.step()
 step(m,opt)
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),path)
  step(m,opt);n=DualStreamNet('C5');other=torch.optim.Adam(n.parameters(),lr=1e-4);ck=torch.load(path,weights_only=True);n.load_state_dict(ck['model']);other.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);step(n,other)
  assert all(torch.equal(x,n.state_dict()[k]) for k,x in m.state_dict().items())
 print(json.dumps(dict(models=records,core_initialization_hashes=hashes,adam_resume_exact=True),indent=2))
if __name__=='__main__':main()
