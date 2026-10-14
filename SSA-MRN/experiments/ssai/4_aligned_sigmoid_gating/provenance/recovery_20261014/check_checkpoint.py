from pathlib import Path
import json,sys,hashlib,torch
root=Path('C:/CtrS-ssai4/SSA-MRN');sys.path.insert(0,str(root))
from common.ssai_suite import model,configure,sha
run=Path('C:/CtrS-ssai4/runtime/qb_s42_100');plan=json.loads((run/'plan.json').read_text());backup=run/'recovery_20261014_1533'
assert all(sha(root/name)==h for name,h in plan['source_hashes'].items())
configure(42,'cpu');rows=[]
for arm,expected in [('A0',72),('A1',73)]:
 history=json.loads((run/arm/'history.json').read_text());assert history[-1]['epoch']==expected
 for name in ('best.pt','latest.pt'):
  p=run/arm/name;assert sha(p)==sha(backup/arm/name)
  ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['fingerprint']==hashlib.sha256(json.dumps({'plan':plan,'arm':arm},sort_keys=True).encode()).hexdigest()
  assert ck['config']['plan']==plan and all(key in ck for key in ['optimizer','torch_rng','cuda_rng','loader_rng','python_rng','numpy_rng'])
  if name=='latest.pt':assert ck['epoch']==expected
  m=model(arm);m.load_state_dict(ck['model'],strict=True);m.eval()
  with torch.inference_mode():assert torch.isfinite(m(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))).all()
  rows.append({'arm':arm,'checkpoint':name,'epoch':ck['epoch'],'sha256':sha(p),'strict_load':True,'finite_inference':True,'resume_fingerprint_verified':True,'optimizer_rng_preserved':True})
(backup/'checkpoint_check.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
