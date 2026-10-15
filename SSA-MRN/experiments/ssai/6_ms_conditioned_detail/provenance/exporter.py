from pathlib import Path
import sys,json,time,shutil,zipfile
import torch
S=Path('C:/CtrS-ssai-detail/SSA-MRN');R=Path('C:/CtrS-ssai-detail/runtime/qb_s42_100_partial_cache');E=Path('C:/CtrS-ssai-detail/evaluation_source');D=R/'final_delivery'
if D.exists():raise FileExistsError('Export already exists')
D.mkdir();sys.path.insert(0,str(S));from common import ssai_suite as suite
plan=json.loads((R/'plan.json').read_text());summary=json.loads((R/'evaluation_final/summary.json').read_text())
for name,h in plan['source_hashes'].items():
 assert suite.sha(S/name)==h and suite.sha(R/'provenance/source'/name)==h,name
records={}
for arm,folder in [('D6','6_ms_conditioned_detail'),('D7','7_channel_relation_detail')]:
 dst=D/folder;dst.mkdir();hist=json.loads((R/arm/'history.json').read_text());assert [h['epoch']for h in hist]==list(range(1,101))
 minimum=min(h['val_mse']for h in hist);best=next(h['epoch']for h in hist if h['val_mse']==minimum);done=json.loads((R/arm/'complete.json').read_text());records[arm]={}
 def copy(src,relative):
  target=dst/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
 for name in ('best','latest'):
  cp=R/arm/(name+'.pt');ck=torch.load(cp,map_location='cpu',weights_only=False)
  assert suite.sha(cp)==done[name+'_sha256'] and ck['epoch']==(best if name=='best'else 100)
  if name=='best':assert ck['validation_mse']==minimum
  assert ck['config']['plan']==plan and ck['config']['variant']==arm
  net=suite.model(arm);net.load_state_dict(ck['model'],strict=True)
  assert all(torch.isfinite(v).all()for v in ck['model'].values())
  assert ck['optimizer']['state'] and all(k in ck for k in ('torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng'))
  records[arm][name]={'sha256':suite.sha(cp),'bytes':cp.stat().st_size,'epoch':ck['epoch'],'validation_mse':ck['validation_mse'],'strict_load':True,'finite':True,'optimizer_rng_preserved':True}
  copy(cp,f'checkpoints/{arm}_QB_s42/{name}.pt')
 for name in ('history.json','complete.json'):copy(R/arm/name,'results/training/'+name)
 shutil.copytree(R/'evaluation_final'/arm,dst/'results/evaluation')
 for name in ('plan.json','benchmark.json','launch.json','queue.json','training_complete.json'):copy(R/name,'provenance/'+name)
 copy(S.parent/'partial_cache_check.json','results/partial_cache_check.json')
 suite.write(dst/'results/checkpoint_verification.json',{'checked':time.time(),'records':records[arm],'scope':'file integrity and checkpoint restoration; not RR/FR metrics'})
 if arm=='D6':
  for name in ('summary.json','complete.json','selection.json','references.json'):copy(R/'evaluation_final'/name,'results/evaluation/'+name)
  with zipfile.ZipFile(dst/'provenance/training_source.zip','w',zipfile.ZIP_DEFLATED)as z:
   for path in(R/'provenance/source').rglob('*'):
    if path.is_file():z.write(path,'source/'+path.relative_to(R/'provenance/source').as_posix())
   z.write(R/'plan.json','plan.json')
  copy(E/'common/ssai_evaluate.py','provenance/evaluator.py');copy(Path(__file__),'provenance/exporter.py')
 else:
  suite.write(dst/'provenance/shared_evidence.json',{'training_source':{'path':'../6_ms_conditioned_detail/provenance/training_source.zip','sha256':suite.sha(D/'6_ms_conditioned_detail/provenance/training_source.zip')},'evaluation_summary':{'path':'../6_ms_conditioned_detail/results/evaluation/summary.json','sha256':suite.sha(D/'6_ms_conditioned_detail/results/evaluation/summary.json')},'evaluator':{'path':'../6_ms_conditioned_detail/provenance/evaluator.py','sha256':suite.sha(D/'6_ms_conditioned_detail/provenance/evaluator.py')},'policy':'shared raw suite evidence referenced without duplicating checkpoint/training runs'})
  suite.write(dst/'results/evaluation/summary.json',{'derived_from':'../6_ms_conditioned_detail/results/evaluation/summary.json','means':{a:summary['means'][a]for a in ('A0','D6','D7')},'difference_from_A0':summary['differences']['D7'],'profiles':{a:summary['profiles'][a]for a in ('A0','D6','D7')},'selection':summary['selection']['D7'],'additional_actual_predictions':40,'protocols':['RR','FR']})
files={p.relative_to(D).as_posix():{'bytes':p.stat().st_size,'sha256':suite.sha(p)}for p in D.rglob('*')if p.is_file()}
suite.write(D/'export_manifest.json',{'files':files})
with zipfile.ZipFile(R/'final_delivery.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in D.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(D))
print(json.dumps({'zip_bytes':(R/'final_delivery.zip').stat().st_size,'zip_sha256':suite.sha(R/'final_delivery.zip'),'files':len(files)+1,'records':records}),flush=True)
