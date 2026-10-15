import sys,json,torch
from pathlib import Path
sys.path.insert(0,'/private/tmp/ctrs-ssai4-delivery/SSA-MRN')
from common import ssai_suite as suite
suite.configure(42,'cpu');base=suite.model('A0');rng=torch.get_rng_state().clone();state=base.state_dict()
p=torch.rand(2,1,64,64);l=torch.rand(2,4,64,64);ms=torch.rand(2,4,16,16);base.eval()
with torch.inference_mode():expected=base(p,l,ms)
records={}
for arm in ('D6','D7'):
 suite.configure(42,'cpu');model=suite.model(arm)
 assert torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(v,model.state_dict()[k])for k,v in state.items())
 model.eval()
 with torch.inference_mode():out=model(p,l,ms)
 torch.testing.assert_close(out,expected,rtol=0,atol=0)
 model.train();out=model(p,l,ms);out.square().mean().backward()
 grads=[m.detail_out.weight.grad for m in model.modules()if hasattr(m,'detail_out')]
 assert all(g is not None and torch.isfinite(g).all()for g in grads)
 assert sum(g.abs().sum().item()for g in grads)>0
 assert torch.isfinite(out).all()
 assert sum(v.numel()for v in model.parameters())==426846
 for shape in ((32,48),(64,64)):
  model.eval();h,w=shape
  with torch.inference_mode():v=model(torch.rand(1,1,h,w),torch.rand(1,4,h,w),torch.rand(1,4,h//4,w//4))
  assert v.shape==(1,4,h,w)and torch.isfinite(v).all()
 records[arm]={'parameters':426846,'new_parameters':192,'original_initial_parameters_equal':True,'original_rng_equal':True,'initial_full_output_exact_equal':True,'finite_backward':True,'nonzero_new_output_gradient':True,'rectangular_inference_passed':True}
Path('/private/tmp/ssai-detail-implementation-check.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
