"""Verify control equivalence, live new paths, final-only PAN and exact Adam recovery."""
import hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
module=importlib.import_module('research.performance.22_stage_aware_restoration.model')
StageAwareNet=module.StageAwareNet;VARIANTS=module.VARIANTS;GuidanceNet=module.GuidanceNet
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 torch.set_num_threads(1);torch.manual_seed(901)
 inp=[torch.rand(2,c,h,w) for c,h,w in [(1,32,48),(4,32,48),(4,8,12)]];gt=torch.rand_like(inp[1]);records={};hashes=[]
 for variant in VARIANTS:
  torch.manual_seed(45);m=StageAwareNet(variant).eval()
  torch.manual_seed(45);parent=GuidanceNet('D0' if variant=='F5' else 'D3').eval()
  with torch.no_grad():
   expected=parent(*inp);initial=m(*inp)
   assert torch.equal(expected,initial),(variant,'initial control differs')
   assert torch.equal(m.forward_stages(*inp)[-1],initial)
  hashes.append(digest(m.core.state_dict()))
  (m(*inp)-gt).square().mean().backward()
  for name,p in m.named_parameters():
   if name in ('stage_alpha','stage_scale','stage_bias','pan_gate'):
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,(variant,name,'inactive')
  if variant=='F4':
   calls=[0];handles=[layer.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1)) for group in m.pan_projection for layer in group]
   m(*inp);assert calls[0]==12,'PAN projection must run only at final stage'
   for h in handles:h.remove()
  m.zero_grad(set_to_none=True)
  with torch.no_grad():
   if hasattr(m,'stage_scale'):m.stage_scale[2].fill_(.02);m.stage_bias[1].fill_(.01)
   if hasattr(m,'stage_alpha'):m.stage_alpha[1]=.2
   if hasattr(m,'pan_gate'):m.pan_gate.fill_(.2)
  if variant not in ('F0','F5'):assert not torch.equal(m(*inp),initial),(variant,'new logic no effect')
  assert torch.equal(m(*inp),m(*inp)),(variant,'state leaks')
  opt=torch.optim.Adam(m.parameters(),lr=1e-4)
  def step(n,o):
   o.zero_grad(set_to_none=True);(n(*inp)-gt).square().mean().backward();o.step()
  step(m,opt)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),p);step(m,opt)
   n=StageAwareNet(variant);other=torch.optim.Adam(n.parameters(),lr=1e-4);ck=torch.load(p,weights_only=True);n.load_state_dict(ck['model'],strict=True);other.load_state_dict(ck['optimizer']);step(n,other)
   assert all(torch.equal(t,n.state_dict()[name]) for name,t in m.state_dict().items()),(variant,'resume mismatch')
  records[variant]=dict(parameters=sum(p.numel() for p in m.parameters()),initial_parent_equivalence=True,active_added_paths=True,adam_resume_exact=True,trace_final_matches=True)
 assert len(set(hashes))==1
 print(json.dumps(dict(models=records,core_initialization_hash=hashes[0],final_stage_only_pan=True),indent=2))
if __name__=='__main__':main()
