from pathlib import Path
import json,hashlib,zipfile,shutil,time,torch
base=Path('C:/CtrS-ssai4');run=base/'runtime/qb_s42_100';ev=run/'evaluation_final';out=run/'final_delivery';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((ev/'complete.json').read_text())['scene_predictions']==160
records=[]
for arm in ('A0','A1','A2','A3'):
 done=json.loads((run/arm/'complete.json').read_text());h=json.loads((run/arm/'history.json').read_text());assert len(h)==100 and [x['epoch']for x in h]==list(range(1,101));cp=out/'checkpoints'/f'{arm}_QB_s42';cp.mkdir(parents=True)
 for name in ('best.pt','latest.pt'):
  p=run/arm/name;assert sha(p)==done[name.split('.')[0]+'_sha256'];ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['config']['variant']==arm
  if name=='latest.pt':assert ck['epoch']==100
  else:assert ck['epoch']==min(h,key=lambda x:(x['val_mse'],x['epoch']))['epoch']
  shutil.copy2(p,cp/name);records.append({'arm':arm,'checkpoint':name,'epoch':ck['epoch'],'sha256':sha(p)})
 dst=out/'results/training'/arm;dst.mkdir(parents=True)
 for name in ('history.json','complete.json'):shutil.copy2(run/arm/name,dst/name)
shutil.copytree(ev,out/'results/evaluation')
shutil.copy2(run/'training_complete.json',out/'results/training_complete.json')
p=out/'provenance';p.mkdir();shutil.copy2(run/'plan.json',p/'plan.json');shutil.copy2(run/'evaluate_ssai4.py',p/'evaluator.py')
with zipfile.ZipFile(p/'training_source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for x in sorted((run/'provenance').rglob('*')):
  if x.is_file():z.write(x,x.relative_to(run/'provenance').as_posix())
(p/'checkpoint_records.json').write_text(json.dumps(records,indent=2)+'\n')
manifest={x.relative_to(out).as_posix():{'bytes':x.stat().st_size,'sha256':sha(x)}for x in sorted(out.rglob('*'))if x.is_file()};(out/'export_manifest.json').write_text(json.dumps({'files':manifest,'checked':time.time(),'actual_checkpoints':8,'scene_predictions':160},indent=2)+'\n')
zip_path=run/'final_delivery.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED)as z:
 for x in sorted(out.rglob('*')):
  if x.is_file():z.write(x,x.relative_to(out).as_posix())
print(json.dumps({'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path),'files':len(manifest),'checkpoints':records}),flush=True)
