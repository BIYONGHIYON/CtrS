"""Evaluate validation-selected retained weights without starting training."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.checkpoint import experiment,load_model,predict
from common.data import SENSOR_MAX
from common.metrics import rr_metrics,fr_metrics

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def panel(a,y,path,sensor,index):
 bands=[4,2,1] if sensor=='WV3' else [2,1,0]
 ref=a['gt'][bands].transpose(1,2,0);lo,hi=np.percentile(ref,[1,99],axis=(0,1))
 def rgb(v):return Image.fromarray((np.clip((v[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
 pan=a['pan'][0];p0,p1=np.percentile(pan,[1,99]);gray=Image.fromarray((np.clip((pan-p0)/max(p1-p0,1e-8),0,1)*255).astype('uint8')).convert('RGB')
 h,w=pan.shape;canvas=Image.new('RGB',(4*w,h+38),'white');d=ImageDraw.Draw(canvas)
 for j,(label,img) in enumerate([('LR MS',rgb(a['ms'])),('PAN',gray),('Prediction',rgb(y)),('GT',rgb(a['gt']))]):
  canvas.paste(img.resize((w,h),Image.Resampling.NEAREST),(j*w,38));d.text((j*w+8,5),label,fill='black')
 d.text((8,23),f'{sensor} RR index {index} | full scene | shared GT 1-99%',fill='black');canvas.save(path)
 return dict(rgb_assumed_bands_zero_based=bands,display_lo=lo.tolist(),display_hi=hi.tolist(),pan_display_lo_hi=[float(p0),float(p1)],no_inference_crop=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=int,choices=(8,9,12),required=True);p.add_argument('--data',type=Path,required=True);args=p.parse_args()
 torch.set_num_threads(2);folder=experiment(args.experiment);out=folder/'results'/'test_evaluation';out.mkdir(exist_ok=True)
 for cp in sorted((folder/'checkpoints').rglob('best.pt')):
  net,ck=load_model(args.experiment,cp);cfg=ck.get('config',{});sensor=cfg.get('sensor','QB');k=cfg.get('k',4);label=cp.parent.name;dest=out/label;dest.mkdir(exist_ok=True)
  for protocol in ('RR','FR'):
   target=dest/(protocol+'.json')
   if target.exists():continue
   name=f'test_{sensor.lower()}'+('_OrigScale' if protocol=='FR' else '')+'_multiExm1.h5';h5=args.data/name;scale=SENSOR_MAX[sensor];rows=[];images=[];start=time.time()
   with h5py.File(h5,'r') as f,torch.inference_mode():
    count=len(f['pan'])
    for i in range(count):
     keys=('pan','lms','ms','gt') if protocol=='RR' else ('pan','lms','ms');a={key:np.asarray(f[key][i],dtype=np.float32) for key in keys}
     x=[torch.from_numpy(a[key]/scale).unsqueeze(0) for key in ('pan','lms','ms')];y=predict(args.experiment,net,*x)[0].numpy().astype(np.float64)*scale
     m=rr_metrics(a['gt'].astype(np.float64),y) if protocol=='RR' else fr_metrics(y,a['lms'],a['ms'],a['pan'])
     if not np.isfinite(y).all() or not all(np.isfinite(v) for v in m.values()):raise ValueError('Non-finite evaluation')
     rows.append(dict(index=i,metrics=m))
     if protocol=='RR' and i in (1,8,11,13,19):
      image=dest/f'scene_{i:02d}.png';display=panel(a,y,image,sensor,i);images.append(dict(index=i,path=image.name,**display))
     print(args.experiment,label,protocol,i+1,count,flush=True)
   report=dict(experiment=args.experiment,variant=label,sensor=sensor,k=k,seed=cfg.get('seed'),checkpoint=str(cp.relative_to(folder)),checkpoint_sha256=digest(cp),epoch=ck.get('epoch'),h5=name,h5_sha256=digest(h5),protocol=protocol,samples=count,rows=rows,mean={key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']},images=images,seconds=time.time()-start,torch=torch.__version__,inference='CPU FP32 full scene; validation-selected best; no clipping or inference crop',limits='QNR/Q2n/SCC MATLAB parity is unverified; RGB band order is assumed.')
   target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
