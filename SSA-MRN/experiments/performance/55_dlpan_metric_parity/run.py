"""Recompute Python metrics against frozen MATLAB results; no training/inference."""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import h5py,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def check(a,b,label):
 a,b=np.asarray(a),np.asarray(b)
 if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(label+' nonfinite')
 if a.shape!=b.shape or not np.allclose(a,b,atol=1e-10,rtol=1e-10):raise ValueError(label+' mismatch')
 print(label, 'PASS',a.size,'max_abs',float(np.max(abs(a-b))))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
 paths={'RR':a.data/'test_qb_multiExm1.h5','FR':a.data/'test_qb_OrigScale_multiExm1.h5'}
 for k,v in paths.items():
  if digest(v)!=cfg['data_sha256'][k]:raise ValueError(k+' data hash mismatch')
 spec=importlib.util.spec_from_file_location('current_metrics',ROOT/'common/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with (HERE/'results/rr/rr_comparison.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 keys=['SAM','ERGAS','Q2n','SCC','PSNR'];actual=[];expected=[]
 with h5py.File(paths['RR']) as f:
  if f['gt'].shape!=(20,4,256,256):raise ValueError('RR shape')
  for i in range(20):
   r=m.rr_metrics(np.asarray(f['gt'][i],dtype=float),np.asarray(f['lms'][i],dtype=float))
   scene='QB_LMS_scene_%02d'%i
   for k in keys:
    label='PSNR_project' if k=='PSNR' else k
    matched=[x for x in rows if x['Case']==scene and x['Metric']==label]
    if len(matched)!=1:raise ValueError('missing/duplicate oracle')
    actual.append(r[k]);expected.append(float(matched[0]['MATLAB']))
 check(actual,expected,'RR')
 oracle=json.loads((HERE/'results/fr_matlab_oracle.json').read_text(encoding='utf-8'))['matlab_scores'];actual=[]
 with h5py.File(paths['FR']) as f:
  if len(f['pan'])!=20:raise ValueError('FR scenes')
  for i in range(20):
   lms=np.asarray(f['lms'][i],dtype=float);ms=np.asarray(f['ms'][i],dtype=float);pan=np.asarray(f['pan'][i],dtype=float).squeeze()
   pf=m._interp23tap(m._matlab_down4(pan),4)
   pred=lms*np.array([.95,1.02,1.05,.98])[:,None,None]+np.array([.1,.2,.3,.4])[:,None,None]*(pan-pf)[None]
   r=m.fr_metrics(pred,lms,ms,pan,pan_resize='matlab');actual.append([r[k] for k in ['D_lambda','D_s','QNR']])
 check(actual,oracle,'FR')
 print('PASS: QB RR 100 + FR 60; frozen MATLAB oracle, not model performance')
if __name__=='__main__':main()
