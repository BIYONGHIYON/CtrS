from pathlib import Path
import json,hashlib,zipfile,shutil,sys,time
import torch
R=Path('C:/CtrS-ssai5/runtime/qb_s42_100');S=Path('C:/CtrS-ssai5/SSA-MRN');E=Path('C:/CtrS-ssai5/evaluation_source');D=R/'final_delivery'
if D.exists():raise FileExistsError('Export exists')
D.mkdir();sys.path.insert(0,str(S));from common import ssai_suite as suite
plan=json.loads((R/'plan.json').read_text())
for name,h in plan['source_hashes'].items():
 if suite.sha(S/name)!=h or suite.sha(R/'provenance/source'/name)!=h:raise ValueError('Training source mismatch: '+name)
records={}
def copy(src,dst):
 dst=D/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for arm in ('M25','M50'):
 history=json.loads((R/arm/'history.json').read_text());assert len(history)==100 and [h['epoch']for h in history]==list(range(1,101))
 minimum=min(x['val_mse']for x in history);epoch=next(x['epoch']for x in history if x['val_mse']==minimum)
 done=json.loads((R/arm/'complete.json').read_text());records[arm]={}
 for name in ('best','latest'):
  cp=R/arm/(name+'.pt');ck=torch.load(cp,map_location='cpu',weights_only=False)
  assert ck['epoch']==(epoch if name=='best'else 100)
  assert suite.sha(cp)==done[name+'_sha256']
  assert ck['config']['plan']==plan and ck['config']['variant']==arm
  m=suite.model(arm);m.load_state_dict(ck['model'],strict=True)
  assert all(torch.isfinite(v).all()for v in ck['model'].values())
  assert ck['optimizer']['state'] and all(k in ck for k in ('torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng'))
  records[arm][name]={'sha256':suite.sha(cp),'bytes':cp.stat().st_size,'epoch':ck['epoch'],'validation_mse':ck['validation_mse'],'strict_load':True,'finite':True,'optimizer_rng_preserved':True}
  copy(cp,Path('checkpoints')/f'{arm}_QB_s42'/(name+'.pt'))
 for name in ('history.json','complete.json'):copy(R/arm/name,Path('results/training')/arm/name)
copy(R/'evaluation_final',Path('not-used')) if False else None
shutil.copytree(R/'evaluation_final',D/'results/evaluation')
for name in ('plan.json','benchmark.json','launch.json','training_complete.json','queue.json'):copy(R/name,Path('provenance')/name)
copy(E/'common/ssai_evaluate.py','provenance/evaluator.py');copy(Path(__file__),'provenance/exporter.py')
copy(R/'startup_check.json','results/startup_check.json')
with zipfile.ZipFile(D/'provenance/training_source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in(R/'provenance/source').rglob('*'):
  if p.is_file():z.write(p,'source/'+p.relative_to(R/'provenance/source').as_posix())
 z.write(R/'plan.json','plan.json')
suite.write(D/'results/checkpoint_verification.json',{'checked':time.time(),'scope':'file hashes, checkpoint state/config, strict loading; separate from RR/FR evaluation','records':records})
files={p.relative_to(D).as_posix():{'bytes':p.stat().st_size,'sha256':suite.sha(p)}for p in D.rglob('*')if p.is_file()}
suite.write(D/'provenance/export_manifest.json',{'files':files})
with zipfile.ZipFile(R/'final_delivery.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in D.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(D))
print(json.dumps({'files':len(files)+1,'zip_bytes':(R/'final_delivery.zip').stat().st_size,'zip_sha256':suite.sha(R/'final_delivery.zip'),'records':records}),flush=True)
