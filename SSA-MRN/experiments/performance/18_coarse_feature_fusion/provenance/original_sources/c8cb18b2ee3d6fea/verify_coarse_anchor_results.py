"""Verify original coarse-anchor weights,300epochs,validation choice and full RR/FR evidence."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=ROOT/'docs/assets/windows_coarse_anchor_qb_50');p.add_argument('--check-torch',action='store_true');p.add_argument('--data',type=Path);a=p.parse_args();r=a.assets
 manifest=read(r/'report_manifest.json')
 for n,x in manifest['artifacts'].items():assert (r/n).stat().st_size==x['bytes'] and sha(r/n)==x['sha256'],n
 for n,h in manifest['delivery_source_hashes'].items():assert sha(ROOT/n)==h,n
 inv=read(r/'server_inventory.json')
 for n,x in inv['artifacts'].items():assert (r/n).stat().st_size==x['bytes'] and sha(r/n)==x['sha256'],n
 plan=read(r/'plan.json');assert (plan['epochs'],plan['effective_batch'],plan['micro'],plan['K'],plan['seed'],plan['workers'])==(50,64,64,4,42,1)
 for n,h in plan['source_hashes'].items():
  assert sha(r/'source_snapshot'/n)==h,n
  if (ROOT/n).exists():assert (ROOT/n).read_bytes().replace(b'\r\n',b'\n')==(r/'source_snapshot'/n).read_bytes().replace(b'\r\n',b'\n'),n
 variants=tuple(plan['variants']);assert variants==('N0','N1','N2','N3','N4','N5')
 selected={}
 for v in variants:
  h=read(r/'runs'/v/'history.json');assert [x['epoch'] for x in h]==list(range(1,51))
  assert all(math.isfinite(row[k]) for row in h for k in ('train_mse','val_mse','val_sam','seconds'))
  best=min(h,key=lambda x:(x['val_mse'],x['val_sam'],x['epoch']));selected[v]=best
  complete=read(r/'runs'/v/'complete.json');assert complete['epochs']==50 and complete['best_mse']==best['val_mse']
  for kind in ('best','latest'):
   assert sha(r/'runs'/v/(kind+'.pt'))==complete[kind+'_sha256']
   assert inv['checkpoints'][v][kind]['epoch']==(best['epoch'] if kind=='best' else 50)
 rank=read(r/'ranking.json')['rows'];expected=[dict(variant=v,best_mse=selected[v]['val_mse'],val_sam=selected[v]['val_sam'],epoch=selected[v]['epoch']) for v in variants];expected.sort(key=lambda x:(x['best_mse'],x['val_sam'],variants.index(x['variant'])));assert rank==expected and rank[0]['variant']=='N3'
 decision=read(r/'combination_decision.json');assert decision['eligible'] and decision['scores']=={v:selected[v]['val_mse'] for v in ('N0','N2','N3','N4')};assert selected['N3']['val_mse']<selected['N2']['val_mse'] and selected['N4']['val_mse']<selected['N0']['val_mse']
 q=read(r/'queue.json');assert not q['active'] and not q['failed'] and not q['skipped'] and set(q['completed'])==set(variants)
 suite=read(r/'suite_complete.json');assert suite['winner']=='N3' and suite['epochs']==50 and suite['runs']==list(variants)
 report=read(r/'evaluation/metrics.json');assert set(report['runs'])=={'N0','N3'}
 assert read(r/'evaluation/selected_models.json')['candidate']=='N3'
 selection=read(r/'evaluation/selection.json');assert selection==report['selection'] and selection['indices']==[1,8,11,13,19]
 for v,item in report['runs'].items():
  assert item['best_epoch']==selected[v]['epoch'] and item['best_sha256']==sha(r/'runs'/v/'best.pt')
  for protocol,record in item['metrics'].items():
   assert [x['index'] for x in record['rows']]==list(range(20))
   for k,m in record['mean'].items():assert math.isclose(sum(x['metrics'][k] for x in record['rows'])/20,m,rel_tol=1e-12,abs_tol=1e-14)
  for i in selection['indices']:assert (r/f'evaluation/{v}_scene_{i:02}.png').exists()
 for n in ('curves.png','gate_curves.png','test_comparison.png'):assert (r/'evaluation'/n).is_file()
 if a.check_torch:
  import torch
  sys.path.insert(0,str(r/'source_snapshot/src'));from ssamrn.models.coarse_anchor import CoarseAnchorNet
  torch.set_num_threads(1)
  for v in variants:
   for kind in ('best','latest'):
    ck=torch.load(r/'runs'/v/(kind+'.pt'),map_location='cpu',weights_only=False);assert ck['config']==dict(plan=plan,variant=v)
    fp=hashlib.sha256(json.dumps(ck['config'],sort_keys=True).encode()).hexdigest();assert ck['fingerprint']==fp==read(r/'runs'/v/'complete.json')['fingerprint']
    assert ck['epoch']==(selected[v]['epoch'] if kind=='best' else 50) and ck['next_batch']==0
    assert all(k in ck for k in ('optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng'))
    m=CoarseAnchorNet(v).eval();m.load_state_dict(ck['model'],strict=True)
    if kind=='best':
     with torch.inference_mode():y=m(torch.ones(1,1,64,64),torch.ones(1,4,64,64),torch.ones(1,4,16,16))
     assert y.shape==(1,4,64,64) and torch.isfinite(y).all()
  if a.data:
   import numpy as np,h5py
   paths=list(a.data.rglob('test_qb_multiExm1.h5'));assert len(paths)==1 and sha(paths[0])==report['runs']['N0']['metrics']['RR']['data_sha256']
   with h5py.File(paths[0]) as f:
    inputs=[torch.from_numpy(np.asarray(f[k][0],dtype=np.float32)/2047.).unsqueeze(0) for k in ('pan','lms','ms')];gt=torch.from_numpy(np.asarray(f['gt'][0],dtype=np.float32)/2047.)
   for v in ('N0','N3'):
    ck=torch.load(r/'runs'/v/'best.pt',map_location='cpu',weights_only=False);m=CoarseAnchorNet(v).eval();m.load_state_dict(ck['model'])
    with torch.inference_mode():y=m(*inputs)[0]
    actual=float((y-gt).double().square().mean());expected=report['runs'][v]['metrics']['RR']['rows'][0]['metrics']['MSE_peak1'];assert math.isclose(actual,expected,rel_tol=1e-4,abs_tol=1e-10),(v,actual,expected)
 print('PASS:79 original server artifacts,12 full checkpoints,300 epochs,N3 validation choice,80 full-scene rows,10 panels;'+('strict loads and inference passed' if a.check_torch else 'hashes verified'))
if __name__=='__main__':main()
