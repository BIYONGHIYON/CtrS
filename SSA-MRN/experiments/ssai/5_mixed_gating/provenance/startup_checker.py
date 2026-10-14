import sys,json,time
from pathlib import Path
import torch
sys.path.insert(0,'C:/CtrS-ssai5/SSA-MRN')
from common import ssai_suite as suite
r=Path('C:/CtrS-ssai5/runtime/qb_s42_100')
out={'checked':time.time(),'queue':json.loads((r/'queue.json').read_text()),'arms':{}}
for arm in ('M25','M50'):
 d={'status':json.loads((r/arm/'status.json').read_text()),'checkpoints':{}}
 if (r/arm/'history.json').exists():d['history']=json.loads((r/arm/'history.json').read_text())
 for name in ('best','latest'):
  p=r/arm/(name+'.pt')
  if p.exists():
   ck=torch.load(p,map_location='cpu',weights_only=False);m=suite.model(arm);m.load_state_dict(ck['model'],strict=True)
   assert all(torch.isfinite(x).all() for x in ck['model'].values())
   d['checkpoints'][name]={'epoch':ck['epoch'],'sha256':suite.sha(p),'strict_load':True,'finite':True,'optimizer_restored':bool(ck['optimizer']['state'])}
 out['arms'][arm]=d
suite.write(r/'startup_check.json',out)
print(json.dumps(out),flush=True)
