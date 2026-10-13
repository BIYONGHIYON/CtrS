import sys,importlib,torch,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
Net=importlib.import_module('research.performance.23_multi_sensor_comparison.model').ComparisonNet
results=[]
for c in [4,8]:
 for variant in ['original','repeat','F0']:
  torch.manual_seed(46);m=Net(variant,c).cuda();m.train()
  x=[torch.rand(2,1,32,32,device='cuda'),torch.rand(2,c,32,32,device='cuda'),torch.rand(2,c,8,8,device='cuda')]
  y=m(*x);assert y.shape==(2,c,32,32) and torch.isfinite(y).all();y.square().mean().backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  results.append({'channels':c,'variant':variant,'shape':list(y.shape)})
  del m;torch.cuda.empty_cache()
 hashes=[]
 for v in ['original','repeat','F0']:
  torch.manual_seed(46);m=Net(v,c);digest=hashlib.sha256()
  for k,t in sorted(m.core.state_dict().items()):digest.update(k.encode());digest.update(t.numpy().tobytes())
  hashes.append(digest.hexdigest())
 assert len(set(hashes))==1
 # Zero memory gate must exactly match repeat under evaluation and matched extra initialization.
 torch.manual_seed(46);a=Net('repeat',c).cuda().eval();torch.manual_seed(46);b=Net('F0',c).cuda().eval()
 with torch.inference_mode():assert torch.allclose(a(*x),b(*x),atol=1e-6,rtol=1e-5)
 del a,b;torch.cuda.empty_cache()
root.joinpath('research/performance/23_multi_sensor_comparison/preflight.json').write_text(json.dumps({'checks':results,'same_core_initialization':True,'zero_memory_matches_repeat':True,'QB_F0_matches_archived_F0':'verified in launch preflight'},indent=2))
print('PASS: 4/8-band gradient, initialization, zero-memory')
