"""Verify control equivalence, zero-gate identity, live new paths and exact Adam recovery."""
import hashlib,json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
GuidanceNet = __import__('importlib').import_module('research.performance.20_recurrent_guidance.model').GuidanceNet; VARIANTS = __import__('importlib').import_module('research.performance.20_recurrent_guidance.model').VARIANTS

def digest(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def inputs():
 torch.manual_seed(907)
 return [torch.rand(2,c,h,w) for c,h,w in [(1,32,48),(4,32,48),(4,8,12)]]
def main():
 torch.set_num_threads(1);inp=inputs();gt=torch.rand_like(inp[1]);models={};records={};outputs={};hashes={};mem={}
 for v in VARIANTS:
  torch.manual_seed(43);m=GuidanceNet(v).eval();models[v]=m;hashes[v]=digest(m.core.state_dict())
  with torch.no_grad():y=m(*inp)
  assert y.shape==(2,4,32,48) and torch.isfinite(y).all();outputs[v]=y
  if v!='D0':
   current=inp[1]
   with torch.no_grad():
    for _ in range(2 if v=='D2' else 3):current=current+m.alpha*(m.core(inp[0],current+m.error_encoder(torch.zeros_like(current)),inp[2])-current)
   assert torch.equal(y,current),(v,'zero-gate control equivalence')
  if hasattr(m,'memory_projection'):mem[v]=digest(m.memory_projection.state_dict())
  if v in ('D3','D4','D5','D6'):
   loss=(m(*inp)-gt).square().mean();loss.backward()
   gate_names=[n for n,_ in m.named_parameters() if n.endswith('_gate')]
   assert all(m.get_parameter(n).grad is not None and m.get_parameter(n).grad.abs().sum()>0 for n in gate_names),(v,'gate unable to open')
   m.zero_grad(set_to_none=True)
   with torch.no_grad():
    for n in gate_names:m.get_parameter(n).fill_(0.2)
   opened=m(*inp);assert not torch.equal(opened,y),(v,'new path inactive')
   (opened-gt).square().mean().backward()
   for n,param in m.named_parameters():
    if n.startswith(('memory_projection','pan_projection')) or n.endswith('_gate'):
     assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0,(v,n)
  else:(m(*inp)-gt).square().mean().backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  with torch.inference_mode():
   pred=m(*inp);no_pan=m(torch.zeros_like(inp[0]),*inp[1:]);no_ms=m(*inp[:2],torch.zeros_like(inp[2]));assert not torch.equal(pred,no_pan) and not torch.equal(pred,no_ms)
   assert torch.equal(pred,m(*inp)),(v,'memory leaked between forwards')
  # Snapshot rather than retain state_dict references across optimizer updates.
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'weights.pt';torch.save(m.state_dict(),p);n=GuidanceNet(v).eval();n.load_state_dict(torch.load(p,weights_only=True),strict=True)
   with torch.inference_mode():assert torch.equal(n(*inp),pred)
  records[v]=dict(parameters=sum(p.numel() for p in m.parameters()),finite_gradients=True,pan_ms_used=True,zero_gate_identity=v!='D0',strict_roundtrip=True,shape=list(pred.shape))
 assert len(set(hashes.values()))==1
 assert len(set(mem.values()))==1
 assert not torch.equal(outputs['D0'],outputs['D1']) and not torch.equal(outputs['D1'],outputs['D2'])
 for v in ('D3','D4','D5','D6'):assert torch.equal(outputs[v],outputs['D1'])
 # Verify each D6 added path independently changes the same trained state.
 m=models['D6']
 with torch.no_grad():
  both=m(*inp);m.pan_gate.zero_();memory_only=m(*inp);m.pan_gate.fill_(0.2);m.memory_gate.zero_();pan_only=m(*inp)
 assert not torch.equal(both,memory_only) and not torch.equal(both,pan_only)
 torch.manual_seed(43);m=GuidanceNet('D6');opt=torch.optim.Adam(m.parameters(),lr=1e-4)
 def step(net,optimizer):optimizer.zero_grad(set_to_none=True);(net(*inp)-gt).square().mean().backward();optimizer.step()
 step(m,opt)
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),p);step(m,opt)
  n=GuidanceNet('D6');other=torch.optim.Adam(n.parameters(),lr=1e-4);ck=torch.load(p,weights_only=True);n.load_state_dict(ck['model'],strict=True);other.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);step(n,other)
  assert all(torch.equal(t,n.state_dict()[name]) for name,t in m.state_dict().items())
 print(json.dumps(dict(models=records,core_initialization_hashes=hashes,memory_initialization_hashes=mem,adam_resume_exact=True,D6_paths_independently_live=True),indent=2))
if __name__=='__main__':main()
