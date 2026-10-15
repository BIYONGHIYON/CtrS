"""Preserve actual training weights, complete histories and evaluation without mutation."""
import argparse,importlib,json,shutil,sys,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--evaluator',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.source_root))
from common import ssai_suite as s
import torch
cls=importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model').MeanNormalizedNet
D=a.runtime/'final_delivery'
if D.exists():raise FileExistsError('Fresh export required')
D.mkdir();plan=json.loads((a.runtime/'plan.json').read_text());records={}
for name,h in plan['source_hashes'].items():
 assert s.sha(a.source_root/name)==h and s.sha(a.runtime/'provenance/source'/name)==h,name
assert json.loads((a.evaluation/'complete.json').read_text())['full_evaluation_complete']
def copy(path,rel):
 dst=D/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
for arm in ['N1','N2']:
 hist=json.loads((a.runtime/arm/'history.json').read_text());assert [x['epoch']for x in hist]==list(range(1,101))
 best=min(hist,key=lambda x:x['val_mse']);done=json.loads((a.runtime/arm/'complete.json').read_text());records[arm]={}
 for kind in ['best','latest']:
  path=a.runtime/arm/(kind+'.pt');ck=torch.load(path,map_location='cpu',weights_only=False)
  assert ck['epoch']==(best['epoch']if kind=='best'else 100) and ck['config']['plan']==plan and ck['config']['variant']==arm
  assert s.sha(path)==done[kind+'_sha256'] and all(torch.isfinite(v).all()for v in ck['model'].values())
  assert ck['optimizer']['state'] and all(k in ck for k in ['torch_rng','cuda_rng','loader_rng','python_rng','numpy_rng'])
  net=cls(learn_gamma=arm=='N2');net.load_state_dict(ck['model'],strict=True)
  records[arm][kind]={'epoch':ck['epoch'],'sha256':s.sha(path),'bytes':path.stat().st_size,'strict_load':True,'optimizer_rng_preserved':True}
  copy(path,f'checkpoints/{arm}_QB_s42/{kind}.pt')
 for name in ['history.json','complete.json']:copy(a.runtime/arm/name,f'results/training/{arm}/{name}')
for name in ['preflight.json','diagnostics.json','benchmark.json','environment_check.json']:copy(a.runtime/name,'results/'+name)
for name in ['plan.json','launch.json','queue.json','training_complete.json']:copy(a.runtime/name,'provenance/'+name)
copy(a.runtime/'provenance/training_source.zip','provenance/training_source.zip')
copy(a.evaluator,'provenance/evaluator.py');copy(Path(__file__),'provenance/exporter.py')
shutil.copytree(a.evaluation,D/'results/evaluation')
s.write(D/'results/checkpoint_verification.json',records)
refs={}
for arm in ['A0','A2']:
 refs[arm]={}
 for kind in ['best','latest']:
  record=plan['reference_runs'][arm][kind]
  refs[arm][kind]={'repository_path':f'experiments/ssai/4_aligned_sigmoid_gating/checkpoints/{arm}_QB_s42/{kind}.pt','sha256':record['sha256']}
 for protocol in ['RR','FR']:
  path=Path(plan['reference_runs'][arm]['best']['path']).parents[1]/'evaluation_final'/arm/(protocol+'.json')
  refs[arm][protocol]={'repository_path':f'experiments/ssai/4_aligned_sigmoid_gating/results/evaluation/{arm}/{protocol}.json','sha256':s.sha(path)}
s.write(D/'provenance/reference_assets.json',{'files':refs})
files={p.relative_to(D).as_posix():{'bytes':p.stat().st_size,'sha256':s.sha(p)}for p in D.rglob('*')if p.is_file()}
s.write(D/'provenance/export_manifest.json',{'files':files})
with zipfile.ZipFile(a.runtime/'final_delivery.zip','w',zipfile.ZIP_DEFLATED)as z:
 for path in D.rglob('*'):
  if path.is_file():z.write(path,path.relative_to(D))
print(json.dumps({'files':len(files)+1,'zip_sha256':s.sha(a.runtime/'final_delivery.zip'),'zip_bytes':(a.runtime/'final_delivery.zip').stat().st_size,'checkpoints':records}),flush=True)
