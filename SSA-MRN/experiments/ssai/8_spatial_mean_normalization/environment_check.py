"""Revalidate A0 after observing a changed host driver; never update weights."""
import sys,json,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from common import ssai_suite as s
import torch
from torch.utils.data import DataLoader
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
plan=s.plan(a);s.configure(42,'cuda');ck=torch.load(plan['reference_runs']['A0']['best']['path'],map_location='cpu',weights_only=False)
m=s.model('A0').cuda();m.load_state_dict(ck['model'],strict=True)
v=DataLoader(s.Cached(Path(plan['cache_root'])/'val'),batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
mse,sam=s.validate(m,v,'cuda');expected=ck['validation_mse']
r={'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'samples':1905,'recorded_mse':expected,'current_mse':mse,'absolute_difference':abs(mse-expected),'sam':sam,'passed':abs(mse-expected)<=1e-10,'limit':'fixed checkpoint validation equivalence does not prove identical full training trajectory across drivers'}
s.write(a.runtime/'environment_check.json',r);print(json.dumps(r),flush=True)
if not r['passed']:raise RuntimeError('Control validation mismatch; do not train')
