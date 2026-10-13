import sys,importlib,torch,json,hashlib
from pathlib import Path
root=Path('C:/CtrS-multi-sensor/SSA-MRN');sys.path.insert(0,str(root))
Net=importlib.import_module('experiments.23_multi_sensor_comparison.model').ComparisonNet
results=[]
for c in [4,8]:
 for variant in ['original','repeat','F0']:
  torch.manual_seed(46);m=Net(variant,c).cuda();m.train()
  x=[torch.rand(2,1,32,32,device='cuda'),torch.rand(2,c,32,32,device='cuda'),torch.rand(2,c,8,8,device='cuda')]
  y=m(*x);assert y.shape==(2,c,32,32) and torch.isfinite(y).all();y.square().mean().backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  digest=hashlib.sha256()
  for k,t in sorted(m.core.state_dict().items()):
   # BN statistics change during forward; compare parameter initialization separately below.
   pass
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
# Actual QB F0 is identical to archived F0 under strict load.
sys.path.insert(0,'C:/CtrS-stage-aware/SSA-MRN')
# Load archived standalone model by file; reuse same common model imports.
path=Path('C:/CtrS-stage-aware/SSA-MRN/experiments/22_stage_aware_restoration/model.py')
spec=importlib.util.spec_from_file_location('archived_stage',path);module=importlib.util.module_from_spec(spec)
# Package namespace includes both checkout experiment directories.
import experiments
experiments.__path__.append('C:/CtrS-stage-aware/SSA-MRN/experiments')
spec.loader.exec_module(module)
a=module.StageAwareNet('F0').cuda().eval();b=Net('F0',4).cuda().eval();b.load_state_dict(a.state_dict(),strict=True)
x=[torch.rand(1,1,32,32,device='cuda'),torch.rand(1,4,32,32,device='cuda'),torch.rand(1,4,8,8,device='cuda')]
with torch.inference_mode():assert torch.equal(a(*x),b(*x))
root.joinpath('experiments/23_multi_sensor_comparison/preflight.json').write_text(json.dumps({'checks':results,'same_core_initialization':True,'zero_memory_matches_repeat':True,'QB_F0_matches_archived_F0':True},indent=2))
print('PASS: 4/8-band gradient, initialization, zero-memory, archived F0 equivalence')
