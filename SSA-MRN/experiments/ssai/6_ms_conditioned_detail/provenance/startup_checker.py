import sys,json,time,subprocess
from pathlib import Path
import torch
sys.path.insert(0,'C:/CtrS-ssai-detail/SSA-MRN')
from common import ssai_suite as s
r=Path('C:/CtrS-ssai-detail/runtime/qb_s42_100_partial_cache')
out={'checked':time.time(),'queue':json.loads((r/'queue.json').read_text()),'arms':{}}
for arm in ('D6','D7'):
 d={'status':json.loads((r/arm/'status.json').read_text()),'checkpoints':{}}
 if(r/arm/'history.json').exists():d['history']=json.loads((r/arm/'history.json').read_text())
 for name in ('best','latest'):
  p=r/arm/(name+'.pt')
  if p.exists():
   ck=torch.load(p,map_location='cpu',weights_only=False);net=s.model(arm);net.load_state_dict(ck['model'],strict=True)
   assert all(torch.isfinite(v).all()for v in ck['model'].values())
   d['checkpoints'][name]={'epoch':ck['epoch'],'sha256':s.sha(p),'strict_load':True,'finite':True,'optimizer_state_present':bool(ck['optimizer']['state']),'detail_weight_abs_sum':sum(v.abs().sum().item()for k,v in ck['model'].items()if k.endswith('detail_out.weight'))}
 out['arms'][arm]=d
query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True);out['gpu_observed']=query.stdout.strip()
s.write(r/'startup_check.json',out);print(json.dumps(out),flush=True)
