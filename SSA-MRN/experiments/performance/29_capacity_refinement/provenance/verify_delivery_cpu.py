import sys,pathlib,json,hashlib,math
root=pathlib.Path('C:/CtrS-p150/SSA-MRN');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/performance/1_capacity_refinement'))
from run import StructureNet,sha,write
import torch,numpy as np
torch.set_num_threads(1)
r=pathlib.Path('C:/CtrS-p150/runtime/QB');plan=json.loads((r/'plan.json').read_text());report={'source_hashes':{},'checkpoints':{},'evaluation':{}}
for p,h in plan['source_hashes'].items():assert sha(root/p)==h,p;report['source_hashes'][p]=h
for v in plan['variants']:
 h=json.loads((r/v/'history.json').read_text());assert len(h)==150
 for name in ['best.pt','latest.pt','best_at_50.pt','best_at_100.pt','best_at_150.pt']:
  path=r/v/name;ck=torch.load(path,map_location='cpu',weights_only=False);m=StructureNet(v);m.configure(plan['calibration']);m.load_state_dict(ck['model'],strict=True);m=m.eval()
  batch=[torch.from_numpy(np.array(np.load(pathlib.Path(plan['cache_root'])/'val'/(k+'.npy'),mmap_mode='r')[:1])) for k in ['pan','lms','ms']]
  with torch.no_grad():assert torch.isfinite(m(*batch)).all()
  report['checkpoints'][v+'/'+name]={'sha256':sha(path),'strict_load':True,'finite_forward':True,'best_epoch':ck['best_epoch']}
  del m,batch;torch.cuda.empty_cache()
 for t in [50,100,150]:
  ck=torch.load(r/v/f'best_at_{t}.pt',map_location='cpu',weights_only=False);chosen=min(h[:t],key=lambda q:(q['val_mse'],q['epoch']));assert ck['best_epoch']==chosen['epoch']
for t in [50,100,150]:
 e=json.loads((r/'evaluation'/str(t)/'metrics.json').read_text())
 for v,a in e['runs'].items():
  assert a['best_sha256']==sha(r/v/f'best_at_{t}.pt')
  for p in ['RR','FR']:
   rows=a['metrics'][p]['rows'];assert len(rows)==20
   for k,mean in a['metrics'][p]['mean'].items():assert math.isfinite(mean) and abs(mean-sum(x['metrics'][k] for x in rows)/20)<1e-10
 report['evaluation'][str(t)]='5 models x RR20/FR20; hashes, means and validation selection verified'
write(pathlib.Path('C:/CtrS-p150/verification.json'),report);print(json.dumps({'sources':len(report['source_hashes']),'checkpoints':len(report['checkpoints']),'milestones':list(report['evaluation']),'pass':True}))
