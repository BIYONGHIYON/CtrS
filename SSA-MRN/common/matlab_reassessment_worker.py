"""Resumable isolated inference jobs for actual MATLAB reference evaluation.

Each checkpoint runs in a fresh interpreter. A failed record is retried; only
complete exports with matching hashes and adapter version are reused.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
VERSION='matlab-reassessment-v2-'+hashlib.sha256((Path(__file__).with_name('SSA-MRN')/'common/matlab_reassessment_models.py' if (Path(__file__).parent/'SSA-MRN').exists() else Path(__file__).with_name('matlab_reassessment_models.py')).read_bytes()+Path(__file__).read_bytes()).hexdigest()[:12]

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8');t.replace(p)

def reusable(record,out):
 if not record.exists():return False
 old=json.loads(record.read_text(encoding='utf8'))
 return old.get('version')==VERSION and old.get('status')=='inference_exported_matlab_pending' and bool(old.get('exports')) and all((out/'queue'/x['file']).exists() and digest(out/'queue'/x['file'])==x['sha256'] and (out/'inputs'/x['input']).exists() and digest(out/'inputs'/x['input'])==x['input_sha256'] for x in old.get('exports',[]))

def run_job(a,job):
 import numpy as np,torch,h5py
 from scipy.io import savemat
 r=a.runtime;sys.path.insert(0,str(r/'SSA-MRN'))
 from common.matlab_reassessment_models import load,predict
 from common.data import SENSOR_MAX
 c=job['checkpoint'];folder=r/job['experiment'];torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 net,ck,context=load(r,folder,r/c['path']);net=net.cuda().eval();sensor=context['sensor'];scale=SENSOR_MAX[sensor];n=context['number'];track=context['track']
 key=hashlib.sha256(c['path'].encode()).hexdigest()[:20];exports=[]
 if track=='variable_scale':
  legacy=net._audit_legacy;label=Path(c['path']).parent.name
  ratios=[4] if legacy else [int(label[3])] if label.startswith('B_x') else [2,3,4]
  protocols=[('RR',ratio) for ratio in ratios] # No invented FR protocol for synthetic multi-ratio training.
 else:protocols=[('RR',4),('FR',4)]
 weight_sha=digest(r/c['path'])
 for proto,ratio in protocols:
  filename=f'test_{sensor.lower()}'+('_OrigScale' if proto=='FR' else '')+'_multiExm1.h5';matches=list(a.data.rglob(filename))
  if len(matches)!=1:raise ValueError(f'Expected one dataset {filename}, found {len(matches)}')
  h5=matches[0];data_sha=digest(h5);rr_factor=2047./scale if track=='performance' and n==23 and proto=='RR' else 1.
  contract=dict(version=VERSION,experiment=job['experiment'],sensor=sensor,protocol=proto,ratio=ratio,scale=scale,rr_factor=rr_factor,data_sha256=data_sha,weight_sha256=weight_sha,synthetic=track=='variable_scale' and not net._audit_legacy)
  contract_id=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:20]
  input_id=hashlib.sha256(json.dumps({k:v for k,v in contract.items() if k not in ('experiment','weight_sha256')},sort_keys=True).encode()).hexdigest()[:20]
  data_file=input_id+'.mat';inputs=r/'v2/inputs'/data_file;scenes=[];source=[];summary={'PSNR':[],'SAM':[]}
  with h5py.File(h5) as f,torch.inference_mode():
   for i in range(len(f['pan'])):
    vals={k:np.asarray(f[k][i],dtype=np.float32) for k in ('pan','lms','ms','gt') if k in f}
    xs={k:torch.from_numpy(v/scale).unsqueeze(0).cuda() for k,v in vals.items()}
    if contract['synthetic']:
     transformed=net._audit_scale_data.synthetic_pair(xs['gt'],xs['pan'],ratio);xs={k:transformed[k] for k in ('pan','lms','ms','gt')}
     vals={k:v[0].cpu().numpy().astype('float64')*scale for k,v in xs.items()}
    y=predict(net,context,*[xs[k] for k in ('pan','lms','ms')],ratio=ratio)[0].cpu().numpy()
    if not np.isfinite(y).all():raise ValueError('Nonfinite prediction')
    scenes.append(dict(index=i,normalized_pred=y.transpose(1,2,0)))
    if proto=='RR':
     pred=y.astype('float64')*scale*rr_factor;gt=vals['gt'].astype('float64')*rr_factor
     mse=np.mean((pred-gt)**2,axis=(1,2));summary['PSNR'].append(float(np.mean(10*np.log10(2047.**2/mse))))
     dot=np.sum(pred*gt,axis=0);norm=np.linalg.norm(pred,axis=0)*np.linalg.norm(gt,axis=0);valid=norm>0
     summary['SAM'].append(float(np.mean(np.arccos(np.clip(dot[valid]/norm[valid],-1,1)))*180/np.pi))
    if not inputs.exists():
     row={k:(v.astype('float64')*rr_factor if k=='gt' else v).transpose(1,2,0) for k,v in vals.items()};row['pan']=row['pan'][:,:,0];source.append(row)
  if not inputs.exists():
   tmp=Path(str(inputs)+'.part');inputs.parent.mkdir(parents=True,exist_ok=True);savemat(str(tmp),dict(scenes=np.array(source,dtype=object)),do_compression=True,appendmat=False);tmp.replace(inputs)
  dest=r/'v2/queue'/f'{key}_{proto}_x{ratio}.mat';tmp=Path(str(dest)+'.part');dest.parent.mkdir(parents=True,exist_ok=True)
  savemat(str(tmp),dict(id=dest.stem,protocol=proto,peak=2047.,ratio=float(ratio),scale=float(scale*rr_factor),data_file=data_file,contract_id=contract_id,scenes=np.array(scenes,dtype=object)),do_compression=True,appendmat=False);tmp.replace(dest)
  exports.append(dict(file=dest.name,sha256=digest(dest),input=data_file,input_sha256=digest(inputs),contract=contract,contract_id=contract_id,scenes=len(scenes),reproduction_mean={k:float(np.mean(v)) for k,v in summary.items() if v}))
 return dict(status='inference_exported_matlab_pending',version=VERSION,checkpoint=c,exports=exports)

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--job',type=Path);p.add_argument('--max-jobs',type=int);a=p.parse_args();r=a.runtime;out=r/'v2';out.mkdir(exist_ok=True)
 if a.job:
  job=json.loads(a.job.read_text(encoding='utf8'));record=Path(job['record'])
  try:write(record,run_job(a,job))
  except Exception as e:write(record,dict(status='error',version=VERSION,checkpoint=job['checkpoint'],error=repr(e),traceback=traceback.format_exc()));raise
  return
 inv=json.loads((r/'inventory.json').read_text(encoding='utf8'));jobs=[dict(experiment=ex['path'],checkpoint=c) for ex in inv['experiments'] for c in ex['candidate_checkpoints']]
 if a.max_jobs:jobs=jobs[:a.max_jobs]
 ok=failed=0
 for idx,job in enumerate(jobs):
  key=hashlib.sha256(job['checkpoint']['path'].encode()).hexdigest()[:20];record=out/'records'/f'{key}.json';jobfile=out/'jobs'/f'{key}.json';reuse=False
  reuse=reusable(record,out)
  if not reuse:
   job['record']=str(record);write(jobfile,job);log=out/'logs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
   with log.open('ab') as f:
    try:
     proc=subprocess.run([sys.executable,'-u',__file__,'--runtime',str(r),'--data',str(a.data),'--job',str(jobfile)],stdout=f,stderr=subprocess.STDOUT,timeout=1800)
     if proc.returncode and not record.exists():write(record,dict(status='error',checkpoint=job['checkpoint'],error=f'process exit {proc.returncode}'))
    except subprocess.TimeoutExpired:write(record,dict(status='error',checkpoint=job['checkpoint'],error='1800s checkpoint deadline'))
  status=json.loads(record.read_text(encoding='utf8'));ok+=status['status']=='inference_exported_matlab_pending';failed+=status['status']!='inference_exported_matlab_pending'
  write(out/'status.json',dict(state='running',processed=idx+1,total=len(jobs),exported=ok,errors=failed,current=job['checkpoint']['path']))
  print(idx+1,len(jobs),ok,failed,flush=True)
 write(out/'status.json',dict(state='scan_finished',total=len(jobs),exported=ok,errors=failed,complete=failed==0))
if __name__=='__main__':main()
