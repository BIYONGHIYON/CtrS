"""Sequential trusted-checkpoint inference queue; never train or overwrite inputs."""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch,h5py
from scipy.io import savemat

def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();r=a.runtime
 sys.path.insert(0,str(r/'SSA-MRN'))
 from common import checkpoint as loader
 from common.data import SENSOR_MAX
 inv=json.loads((r/'inventory.json').read_text(encoding='utf8'));(r/'queue').mkdir(exist_ok=True);(r/'inputs').mkdir(exist_ok=True);(r/'records').mkdir(exist_ok=True)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 jobs=[(ex,c) for ex in inv['experiments'] for c in ex['candidate_checkpoints']];jobs.sort(key=lambda x: (0 if '/44_' in x[1]['path'] else 1,x[1]['path']));ok=failed=0
 for idx,(ex,c) in enumerate(jobs):
  key=hashlib.sha256(c['path'].encode()).hexdigest()[:20];record=r/'records'/f'{key}.json'
  if record.exists():continue
  start=time.time();folder=Path(ex['path']);number=int(folder.name.split('_')[0]);track=folder.parts[2]
  try:
   if track=='variable_scale' or track=='performance' and number in (23,34):raise ValueError('Original input/scale contract requires specialized adapter; no generic substitution allowed')
   cp=r/c['path'];ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck.get('config',{});plan=cfg.get('plan',{});sensor=cfg.get('sensor',plan.get('sensor',ck.get('sensor','QB')));scale=SENSOR_MAX[sensor]
   if track=='performance' and number in (43,44):
    spec=importlib.util.spec_from_file_location('audit_candidate',r/folder/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);net=mod.Model(ck['arm'],ck.get('seed',42));net.load_state_dict(ck['model'],strict=True)
   else:
    loader.RESEARCH_TRACK=track;net,ck=loader.load_model(number,cp)
   net=net.cuda().eval();protocols=[]
   for proto in ('RR','FR'):
    name=f'test_{sensor.lower()}'+('_OrigScale' if proto=='FR' else '')+'_multiExm1.h5';matches=list(a.data.rglob(name))
    if len(matches)!=1:raise ValueError(f'Expected one dataset {name}, found {len(matches)}')
    h5=matches[0];data_file=f'{sensor}_{proto}.mat';inputs=r/'inputs'/data_file;scenes=[];source=[]
    with h5py.File(h5) as f,torch.inference_mode():
     for i in range(len(f['pan'])):
      vals={k:np.asarray(f[k][i],dtype=np.float32) for k in ('pan','lms','ms','gt') if k in f}
      xs=[torch.from_numpy(vals[k]/scale).unsqueeze(0).cuda() for k in ('pan','lms','ms')]
      y=(net(*xs) if number in (43,44) and track=='performance' else loader.predict(number,net,*xs))[0].cpu().numpy()
      if not np.isfinite(y).all():raise ValueError('Nonfinite prediction')
      scenes.append(dict(index=i,normalized_pred=y.transpose(1,2,0)))
      if not inputs.exists():
       row={k:v.transpose(1,2,0) for k,v in vals.items()};row['pan']=row['pan'][:,:,0];source.append(row)
    if not inputs.exists():
     tmp=Path(str(inputs)+'.part');savemat(str(tmp),dict(scenes=np.array(source,dtype=object)),do_compression=True,appendmat=False);tmp.replace(inputs)
    dest=r/'queue'/f'{key}_{proto}.mat';tmp=Path(str(dest)+'.part');savemat(str(tmp),dict(id=f'{key}_{proto}',protocol=proto,peak=2047.,ratio=4.,scale=float(scale),data_file=data_file,scenes=np.array(scenes,dtype=object)),do_compression=True,appendmat=False);tmp.replace(dest);protocols.append(proto)
   ok+=1;write(record,dict(status='inference_exported_matlab_pending',checkpoint=c,protocols=protocols,seconds=time.time()-start))
  except Exception as e:
   failed+=1;write(record,dict(status='adapter_error_requires_resolution',checkpoint=c,error=repr(e),traceback=traceback.format_exc()))
  finally:
   if 'net' in locals():del net
   torch.cuda.empty_cache()
  write(r/'status.json',dict(state='running',processed=idx+1,total=len(jobs),exported=ok,adapter_errors=failed,current=c['path']))
  print(idx+1,len(jobs),ok,failed,flush=True)
 write(r/'status.json',dict(state='inference_scan_finished',total=len(jobs),exported=ok,adapter_errors=failed,complete=failed==0))
if __name__=='__main__':main()
