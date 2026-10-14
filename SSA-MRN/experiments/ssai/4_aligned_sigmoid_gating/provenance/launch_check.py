"""Read-only CPU checkpoint/load verification after the first saved epoch."""
from pathlib import Path
import sys,json,subprocess,time,hashlib
import torch
root=Path('C:/CtrS-ssai4/SSA-MRN');run=Path('C:/CtrS-ssai4/runtime/qb_s42_100');sys.path.insert(0,str(root))
from common.ssai_suite import model,configure
configure(42,'cpu');checks=[]
for arm in ('A0','A1'):
 for name in ('best.pt','latest.pt'):
  p=run/arm/name;ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['epoch']>=1 and ck['config']['plan']['micro_batch']==32 and ck['config']['plan']['cache_mode']=='gpu'
  net=model(arm);net.load_state_dict(ck['model'],strict=True);net.eval()
  with torch.inference_mode():
   pred=net(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16));assert torch.isfinite(pred).all()
  checks.append({'arm':arm,'checkpoint':name,'epoch':ck['epoch'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'strict_load':True,'finite_cpu_inference':True,'validation_mse':ck['validation_mse']})
queue=json.loads((run/'queue.json').read_text());assert not queue['failed'] and set(queue['active'])=={'A0','A1'}
result={'checked_utc_timestamp':time.time(),'checkpoint_checks':checks,'queue':queue,'status':{a:json.loads((run/a/'status.json').read_text())for a in ('A0','A1')},'gpu':subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()}
(run/'launch_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
