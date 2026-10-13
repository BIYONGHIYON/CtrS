"""Check parent controls, active added paths, gradients and exact Adam resume."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from ssamrn.models.guidance_refinement import RefinementNet,VARIANTS,PARENTS
from ssamrn.models.recurrent_guidance import GuidanceNet

def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 torch.set_num_threads(1);torch.manual_seed(907)
 inp=[torch.rand(2,c,h,w) for c,h,w in [(1,32,48),(4,32,48),(4,8,12)]];gt=torch.rand_like(inp[1]);records={};hashes=[]
 for v in VARIANTS:
  torch.manual_seed(44);m=RefinementNet(v).eval()
  torch.manual_seed(44);parent=GuidanceNet(PARENTS[v]).eval()
  with torch.no_grad():y=m(*inp);expected=parent(*inp)
  assert torch.equal(y,expected),(v,'initial output differs from parent')
  hashes.append(digest(m.core.state_dict()))
  loss=(m(*inp)-gt).square().mean();loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  for n,p in m.named_parameters():
   if n.endswith('_gate') or n.startswith('spatial_gate'):assert p.grad is not None and p.grad.abs().sum()>0,(v,n,'cannot open')
  m.zero_grad(set_to_none=True)
  with torch.no_grad():
   for n,p in m.named_parameters():
    if n.endswith('_gate'):p.fill_(0.2)
   if v=='E5':
    for layer in m.spatial_gate:layer.weight.fill_(.01)
  opened=m(*inp)
  if v in ('E5','E6','E7'):
   with torch.no_grad():parent.load_state_dict({k:t for k,t in m.state_dict().items() if not k.startswith('spatial_gate')},strict=True);old=parent(*inp)
   assert not torch.equal(opened,old),(v,'new logic has no effect')
  (opened-gt).square().mean().backward()
  for n,p in m.named_parameters():
   if n.startswith(('memory_projection','pan_projection','spatial_gate')):assert p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all(),(v,n)
  with torch.no_grad():assert torch.equal(m(*inp),m(*inp)),(v,'state leaks across inputs')
  opt=torch.optim.Adam(m.parameters(),lr=1e-4)
  def step(net,o):o.zero_grad(set_to_none=True);(net(*inp)-gt).square().mean().backward();o.step()
  step(m,opt)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),p);step(m,opt)
   n=RefinementNet(v);other=torch.optim.Adam(n.parameters(),lr=1e-4);ck=torch.load(p,weights_only=True);n.load_state_dict(ck['model'],strict=True);other.load_state_dict(ck['optimizer']);step(n,other)
   assert all(torch.equal(t,n.state_dict()[name]) for name,t in m.state_dict().items()),(v,'Adam resume differs')
  records[v]=dict(parent=PARENTS[v],parameters=sum(p.numel() for p in m.parameters()),initial_parent_equivalence=True,finite_live_gradients=True,adam_resume_exact=True)
 assert len(set(hashes))==1
 assert torch.equal(RefinementNet.highpass(torch.ones(1,4,8,12)),torch.zeros(1,4,8,12))
 print(json.dumps(dict(models=records,core_initialization_hash=hashes[0],constant_highpass_zero=True),indent=2))
if __name__=='__main__':main()
