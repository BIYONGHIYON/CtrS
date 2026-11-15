from pathlib import Path
import sys,json,torch,hashlib
root=Path(__file__).parent/'SSA-MRN';sys.path.insert(0,str(root))
from common.checkpoint import load_model
rt=Path('C:/CtrS-multi-sensor/runtime/seed46');torch.set_num_threads(1);rows=[]
for sensor,channels in [('QB',4),('GF2',4),('WV3',8)]:
 for v in ('original','repeat','F0'):
  for name in ('best.pt','latest.pt','best_at_50.pt','best_at_100.pt','best_at_150.pt'):
   p=rt/sensor/v/name;m,ck=load_model(23,p)
   with torch.inference_mode():y=m(torch.ones(1,1,32,32),torch.ones(1,channels,32,32),torch.ones(1,channels,8,8))
   assert tuple(y.shape)==(1,channels,32,32) and torch.isfinite(y).all()
   if name=='latest.pt':assert ck['epoch']==150 and ck['next_batch']==0 and len(ck['history'])==150 and all(k in ck for k in ('optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng'))
   rows.append({'sensor':sensor,'variant':v,'checkpoint':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'epoch':ck['epoch'],'shape':list(y.shape),'strict_load':True,'finite_forward':True})
report={'checks':rows,'scope':'45 retained original training checkpoints; updated portable checkpoint loader; CPU FP32 32x32 finite forward; not full test re-evaluation'}
Path(__file__).with_name('checkpoint_verification.json').write_text(json.dumps(report,indent=2));print('PASS: 45 strict loads and finite 4/8-band FP32 forward; 9 latest full resume states at epoch150')
