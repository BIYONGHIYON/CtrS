"""Verify frozen recurrent-guidance checkpoints, validation selection and full-scene evidence."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('D0','D1','D2','D3','D4','D5','D6')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=ROOT/'docs/assets/windows_recurrent_guidance_qb_50');p.add_argument('--check-torch',action='store_true');a=p.parse_args();r=a.assets
 original=read(r/'preservation.json')['files'];manifest=read(r/'report_manifest.json')
 for item in original:
  path=r/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
 for name,item in manifest['artifacts'].items():
  path=r/name;assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
 for name,h in manifest['delivery_source_hashes'].items():assert sha(ROOT/name)==h,name
 plan=read(r/'plan.json');assert tuple(plan['variants'])==VARIANTS
 assert (plan['epochs'],plan['effective_batch'],plan['micro'],plan['seed'],plan['workers'],plan['max_parallel'])==(50,96,96,43,1,2)
 for name,h in plan['source_hashes'].items():
  assert sha(r/'source_snapshot'/name)==h,name
 selected={}
 for v in VARIANTS:
  rows=read(r/v/'history.json');assert [x['epoch'] for x in rows]==list(range(1,51))
  assert all(math.isfinite(x[k]) for x in rows for k in ('train_mse','val_mse','val_sam','val_psnr_band_mean','seconds'))
  assert all(x['skipped_steps']==0 for x in rows)
  selected[v]=min(rows,key=lambda x:(x['val_mse'],x['val_sam'],x['epoch']))
  done=read(r/v/'complete.json');assert done['epochs']==50 and done['best_mse']==selected[v]['val_mse']
  for kind in ('best','latest'):assert sha(r/v/(kind+'.pt'))==done[kind+'_sha256']
  status=read(r/v/'status.json');assert status['state']=='complete' and status['epoch']==50
 rank=read(r/'ranking.json')['rows'];expected=[dict(variant=v,best_mse=selected[v]['val_mse'],val_sam=selected[v]['val_sam'],epoch=selected[v]['epoch']) for v in VARIANTS];expected.sort(key=lambda x:(x['best_mse'],x['val_sam'],VARIANTS.index(x['variant'])));assert rank==expected and rank[0]['variant']=='D3'
 candidate=next(x['variant'] for x in rank if x['variant'] not in ('D0','D1'));assert candidate=='D3'
 q=read(r/'queue.json');assert not q['active'] and not q['failed'] and not q['skipped'] and set(q['completed'])==set(VARIANTS)
 done=read(r/'suite_complete.json');assert done['winner']=='D3' and done['runs']==list(VARIANTS) and done['epochs']==50
 report=read(r/'evaluation/metrics.json');assert set(report['runs'])=={'D0','D1','D3'}
 assert read(r/'evaluation/selected_models.json')['candidate']==candidate
 selection=read(r/'evaluation/selection.json');assert selection==report['selection'] and selection['indices']==[1,8,11,13,19] and selection['tile']=='full scene' and selection['frozen_before_test']
 from PIL import Image
 for v,item in report['runs'].items():
  assert item['best_epoch']==selected[v]['epoch'] and item['best_sha256']==sha(r/v/'best.pt')
  assert set(item['metrics'])=={'RR','FR'}
  for protocol,record in item['metrics'].items():
   assert [x['index'] for x in record['rows']]==list(range(20))
   for k,m in record['mean'].items():assert all(math.isfinite(x['metrics'][k]) for x in record['rows']) and math.isclose(sum(x['metrics'][k] for x in record['rows'])/20,m,rel_tol=1e-12,abs_tol=1e-14)
  for i in selection['indices']:
   with Image.open(r/f'evaluation/{v}_scene_{i:02}.png') as im:assert im.width>=1024 and im.height>=256;im.verify()
 for n in ('curves.png','gate_curves.png','test_comparison.png'):
  with Image.open(r/'evaluation'/n) as im:im.verify()
 hashes={v:read(r/v/'initialization.json')['core_sha256'] for v in VARIANTS};assert len(set(hashes.values()))==1
 if a.check_torch:
  import torch
  sys.path.insert(0,str(r/'source_snapshot/src'));from ssamrn.models.recurrent_guidance import GuidanceNet
  torch.set_num_threads(1)
  for v in VARIANTS:
   for kind in ('best','latest'):
    ck=torch.load(r/v/(kind+'.pt'),map_location='cpu',weights_only=False);assert ck['config']==dict(plan=plan,variant=v)
    fp=hashlib.sha256(json.dumps(ck['config'],sort_keys=True).encode()).hexdigest();assert ck['fingerprint']==fp==read(r/v/'complete.json')['fingerprint']
    assert ck['epoch']==(selected[v]['epoch'] if kind=='best' else 50) and ck['next_batch']==0
    assert ck['skipped_steps']==0
    if kind=='best':assert ck['best_epoch']==selected[v]['epoch']
    assert ck['best_mse']==selected[v]['val_mse'] and ck['history']==read(r/v/'history.json')[:ck['epoch']]
    assert all(k in ck for k in ('optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','history'))
    model=GuidanceNet(v).eval();model.load_state_dict(ck['model'],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=plan['lr']);optimizer.load_state_dict(ck['optimizer'])
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
    assert sum(x.numel() for x in model.parameters())==read(r/v/'initialization.json')['parameters']
    with torch.inference_mode():y=model(torch.ones(1,1,64,64),torch.ones(1,4,64,64),torch.ones(1,4,16,16))
    assert y.shape==(1,4,64,64) and torch.isfinite(y).all()
 print(f'PASS: {len(original)} original artifacts; 14 full checkpoints, 350 epochs, D3 validation selection, 120 RR/FR rows, 15 panels; '+('strict loads and finite inference passed' if a.check_torch else 'hashes verified'))
if __name__=='__main__':main()
