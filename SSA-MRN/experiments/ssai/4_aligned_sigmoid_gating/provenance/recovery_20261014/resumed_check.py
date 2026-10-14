from pathlib import Path
import sys,json,time,hashlib,torch,subprocess
root=Path('C:/CtrS-ssai4/SSA-MRN');sys.path.insert(0,str(root))
from common.ssai_suite import configure,model,sha
configure(42,'cpu');r=Path('C:/CtrS-ssai4/runtime/qb_s42_100');plan=json.loads((r/'plan.json').read_text());checks=[]
for arm,ep in [('A0',73),('A1',74)]:
 cp=r/arm/'latest.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']>=ep
 assert ck['fingerprint']==hashlib.sha256(json.dumps({'plan':plan,'arm':arm},sort_keys=True).encode()).hexdigest()
 m=model(arm);m.load_state_dict(ck['model'],strict=True);m.eval()
 with torch.inference_mode():assert torch.isfinite(m(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))).all()
 checks.append({'arm':arm,'resumed_epoch_completed':ep,'latest_epoch':ck['epoch'],'sha256':sha(cp),'strict_load':True,'finite_inference':True,'source_plan_unchanged':True})
q=json.loads((r/'queue.json').read_text());assert not q['failed']
d={'checked_timestamp':time.time(),'checks':checks,'queue':q,'status':{a:json.loads((r/a/'status.json').read_text())for a in ('A0','A1')},'gpu':subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()}
(r/'recovery_20261014_1533/resumed_check.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
