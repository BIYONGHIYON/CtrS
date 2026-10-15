"""Verify retained artifacts and recover four real checkpoints; no metric evaluation."""
import argparse,hashlib,importlib,json,sys,tempfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path);a=p.parse_args();root=a.root.resolve();e=root/'experiments/ssai/8_spatial_mean_normalization'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
manifest=json.loads((e/'manifest.json').read_text())
for name,r in manifest['files'].items():
 path=e/name
 assert path.is_file()and path.stat().st_size==r['bytes']and sha(path)==r['sha256'],name
refs=json.loads((e/'provenance/reference_assets.json').read_text())['files'];ref_count=0
for files in refs.values():
 for r in files.values():assert sha(root/r['repository_path'])==r['sha256'];ref_count+=1
with tempfile.TemporaryDirectory(prefix='ssai8-restore-')as temp:
 t=Path(temp)
 with zipfile.ZipFile(e/'provenance/training_source.zip')as z:
  plan=json.loads(z.read('plan.json'));assert plan==json.loads((e/'provenance/plan.json').read_text())
  for name,h in plan['source_hashes'].items():
   content=z.read('source/'+name);assert hashlib.sha256(content).hexdigest()==h,name
  upstream=t/'references/upstream/network.py';upstream.parent.mkdir(parents=True);upstream.write_bytes(z.read('source/references/upstream/network.py'))
  # Current submitted model code + current baseline adapter, actual upstream source.
  names=['common/model.py','experiments/ssai/8_spatial_mean_normalization/model.py','experiments/ssai/8_spatial_mean_normalization/diagnostics.py']
  for name in names:
   assert sha(root/name)==plan['source_hashes'][name],name
   dest=t/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
 dest=t/'common/checkpoint.py';dest.write_bytes((root/'common/checkpoint.py').read_bytes())
 sys.path.insert(0,str(t));import torch
 torch.set_num_threads(1)
 cls=importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model').MeanNormalizedNet
 diag=importlib.import_module('experiments.ssai.8_spatial_mean_normalization.diagnostics')
 records={};success={};baseline=json.loads((e/'results/diagnostics.json').read_text())['A0_latest_weight_relative_change']
 for arm in ['N1','N2']:
  history=json.loads((e/f'results/training/{arm}/history.json').read_text());assert [x['epoch']for x in history]==list(range(1,101))
  best=min(history,key=lambda x:x['val_mse']);initial=torch.load(e/'provenance/initial_state.pt',map_location='cpu',weights_only=False)['model'];assert diag.state_hash(initial)==plan['initial_common_sha256']
  for kind in ['best','latest']:
   path=e/f'checkpoints/{arm}_QB_s42/{kind}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
   assert ck['epoch']==(best['epoch']if kind=='best'else 100)and ck['config']['variant']==arm and ck['config']['plan']==plan
   if kind=='best':assert ck['validation_mse']==best['val_mse']
   assert ck['optimizer']['state']and all(k in ck for k in ['torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng'])
   assert all(torch.isfinite(v).all()for v in ck['model'].values())
   from common import checkpoint
   checkpoint.RESEARCH_TRACK='ssai'
   net,loaded=checkpoint.load_model(8,path);assert loaded['epoch']==ck['epoch'];net.eval()
   with torch.inference_mode():
    pred=net(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))
   assert pred.shape==(1,4,64,64)and torch.isfinite(pred).all()
   if kind=='latest':
    changes=diag.weight_changes(net,initial)
    for key,value in changes.items():assert abs(value-history[-1]['weight_relative_change'][key])<1e-12
    success[arm]=all(changes[key]>=.1 and changes[key]>=2*baseline[key]for key in diag.STAGES)
   records[arm+'/'+kind]={'sha256':sha(path),'epoch':ck['epoch'],'strict_load':True,'finite_cpu_inference':True,'optimizer_rng_preserved':True}
 result={'verified':True,'manifest_files':len(manifest['files']),'reference_files':ref_count,'checkpoints':records,'learning_success_recomputed_from_latest_weights':success,'scope':'integrity, actual source and checkpoint recovery; no RR/FR recomputation'}
 if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
