import pathlib,json,hashlib,sys,importlib.util,math
import torch,numpy as np
from PIL import Image
root=pathlib.Path(sys.argv[1]);e=root/'SSA-MRN/experiments/28_residual_refinement';snap=e/'provenance/source_snapshot';sys.path.insert(0,str(snap));torch.set_num_threads(2)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
frozen=load('frozen_refiner',snap/'experiments/26_residual_refinement/model.py');current=load('current_refiner',e/'model.py')
plan=json.loads((e/'results/QB/plan.json').read_text());mapping=json.loads((e/'provenance/archive_mapping.json').read_text());export=json.loads((e/'provenance/export_manifest.json').read_text())
for n,d in export.items():b=(e/mapping[n]).read_bytes();assert len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256'],n
for n,digest in plan['source_hashes'].items():assert hashlib.sha256((snap/n).read_bytes()).hexdigest()==digest,n
controls=json.loads((e/'results/QB/controls.json').read_text());new=json.loads((e/'results/QB/evaluation/100/metrics.json').read_text())['runs'];checks=[]
for k,d in controls.items():
 f=root/'SSA-MRN/experiments/23_multi_sensor_comparison/checkpoints'/f"{d['variant']}_QB_s46"/'best_at_100.pt';assert hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],k
for v,d in {**{k:x['result'] for k,x in controls.items()},**new}.items():
 for proto,x in d['metrics'].items():
  assert len(x['rows'])==20 and [r['index'] for r in x['rows']]==list(range(20)),(v,proto)
  for k,mean in x['mean'].items():assert math.isfinite(mean) and abs(mean-np.mean([r['metrics'][k] for r in x['rows']]))<1e-10
for v in ('G3','G4','G5'):
 h=json.loads((e/f'results/QB/{v}/history.json').read_text());assert [x['epoch'] for x in h]==list(range(1,101));assert json.loads((e/f'results/QB/{v}/initialization.json').read_text())['core_sha256']==plan['control_core_sha256']
 complete=json.loads((e/f'results/QB/{v}/complete_100.json').read_text())
 for name in ('best.pt','latest.pt'):
  p=e/f'checkpoints/{v}_QB_s46'/name;c=torch.load(p,map_location='cpu',weights_only=False);assert c['config']['variant']==v and c['config']['plan']==plan and c['best_epoch']==new[v]['best_epoch'];assert hashlib.sha256(p.read_bytes()).hexdigest()==complete['best_sha256' if name=='best.pt' else 'latest_sha256']
  for t in c['model'].values():assert torch.isfinite(t).all()
  for state in c['optimizer']['state'].values():
   for t in state.values():
    if isinstance(t,torch.Tensor):assert torch.isfinite(t).all()
  if name=='latest.pt':assert c['epoch']==100 and c['next_batch']==0 and len(c['history'])==100;assert all(k in c for k in ('python_rng','numpy_rng','torch_rng','cuda_rng','scaler'))
  a=frozen.RefinementNet(v,4);a.configure(plan['calibration']);a.load_state_dict(c['model'],strict=True);a.eval();b=current.RefinementNet(v,4);b.configure(plan['calibration']);b.load_state_dict(c['model'],strict=True);b.eval()
  opt=torch.optim.Adam(b.parameters(),lr=1e-4);opt.load_state_dict(c['optimizer']);torch.manual_seed(46);inputs=(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))
  with torch.inference_mode():y=a(*inputs);z=b(*inputs)
  assert torch.isfinite(y).all() and y.shape==(1,4,64,64) and torch.equal(y,z)
  checks.append({'variant':v,'file':name,'epoch':c['epoch'],'best_epoch':c['best_epoch'],'strict_load':True,'frozen_current_output_identical':True,'skipped_steps':c.get('skipped_steps')})
images=list((e/'results/QB/evaluation/100').glob('*.png'))
for f in images:
 with Image.open(f) as im:im.verify()
print(json.dumps({'export_files':len(export),'checkpoints':checks,'metric_runs':6,'scene_rows':240,'valid_pngs':len(images),'source_hashes':len(plan['source_hashes']),'scope':'hashes, checkpoint load/optimizer/RNG integrity, synthetic CPU inference equivalence; not a new full test evaluation'},indent=2))
