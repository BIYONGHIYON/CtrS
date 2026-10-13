"""Re-evaluate retained public-code reproduction against the multi-sensor suite.

The original checkpoints and their selection remain unchanged. This is an
observational comparison across training recipes, not a controlled ablation.
"""
from pathlib import Path
import argparse,sys,json,hashlib,time,math
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sensors',nargs='+',default=['QB','GF2','WV3']);p.add_argument('--checkpoint-root',type=Path);p.add_argument('--label',default='exp1 epoch100');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 import torch,numpy as np,h5py
 from PIL import Image,ImageDraw
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from common.checkpoint import load_model
 from common.metrics import rr_metrics,fr_metrics
 assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a.output.mkdir(parents=True);report={'scope':a.label+' vs experiment23 validation best; different training recipe/seed/batch','selection':'minimum validation MSE' if a.checkpoint_root else 'historical latest','limitations':['QNR/Q2n/SCC MATLAB parity unverified','GF2 PSNR uses sensor peak1023; historical experiment1 peak2047 PSNR is not directly comparable'],'sensors':{}}
 expected=json.loads((Path(__file__).parent/'manifest.json').read_text())['files']
 for sensor,folder,peak in [('QB','qb',2047.),('GF2','gf2',1023.),('WV3','wv3',2047.)]:
  if sensor not in a.sensors:continue
  out=a.output/sensor;out.mkdir();ckpath=Path(__file__).parent/f'checkpoints/{folder}/latest.pt';name=f'checkpoints/{folder}/latest.pt';
  if a.checkpoint_root:
   ckpath=a.checkpoint_root/sensor/'best.pt';assert sha(ckpath)==json.loads((a.checkpoint_root/sensor/'complete.json').read_text())['best_sha256']
  else:assert sha(ckpath)==expected[name]['sha256']
  net,ck=load_model(1,ckpath);net.cuda();manifest=json.loads((a.reference_runtime/sensor/'data_manifest.json').read_text());reference={str(t):json.loads((a.reference_runtime/sensor/'evaluation'/str(t)/'metrics.json').read_text())['runs'] for t in (100,150)}
  record={'checkpoint_sha256':sha(ckpath),'epoch':ck.get('epoch'),'config':ck.get('config',{}),'ssai_dimension':ck.get('ssai_dimension'),'sensor_peak':peak,'protocols':{},'reference':{}}
  for protocol in ('RR','FR'):
   data=manifest[protocol];h5=Path(data['path']);assert sha(h5)==data['sha256'];rows=[]
   with h5py.File(h5) as f:
    for i in range(len(f['pan'])):
     arrays={k:np.asarray(f[k][i],dtype=np.float32) for k in f};inputs=[torch.from_numpy(arrays[k]/peak).unsqueeze(0).cuda() for k in ('pan','lms','ms')]
     with torch.inference_mode():pred=net(*inputs)[0].cpu().numpy().astype(np.float64)*peak
     metrics=rr_metrics(arrays['gt'].astype(np.float64)*2047./peak,pred*2047./peak) if protocol=='RR' else fr_metrics(pred,arrays['lms'],arrays['ms'],arrays['pan'])
     assert all(math.isfinite(x) for x in metrics.values());rows.append({'index':i,'metrics':metrics})
     if protocol=='RR' and i in [1,8,11,13,19]:
      bands=[4,2,1] if sensor=='WV3' else [2,1,0];ref=arrays['gt'][bands].transpose(1,2,0);lo,hi=np.percentile(ref,[1,99],axis=(0,1))
      def rgb(x):return Image.fromarray((np.clip((x[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
      pn=arrays['pan'][0];a0,a1=np.percentile(pn,[1,99]);pi=Image.fromarray((np.clip((pn-a0)/max(a1-a0,1e-8),0,1)*255).astype('uint8')).convert('RGB');h,w=pn.shape;canvas=Image.new('RGB',(4*w,h+38),'white');draw=ImageDraw.Draw(canvas)
      for j,(label,img) in enumerate([('LR MS',rgb(arrays['ms'])),('PAN',pi),(a.label,rgb(pred)),('GT',rgb(arrays['gt']))]):canvas.paste(img.resize((w,h),Image.Resampling.NEAREST),(j*w,38));draw.text((j*w+8,8),label,fill='black')
      canvas.save(out/f'scene_{i:02}.png')
     print(sensor,protocol,i+1,len(f['pan']),flush=True)
   record['protocols'][protocol]={'h5_sha256':data['sha256'],'rows':rows,'mean':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
  for t,runs in reference.items():
   record['reference'][t]={}
   for v,x in runs.items():
    d={k:x['metrics'][pr]['mean'][k]-record['protocols'][pr]['mean'][k] for pr,k in [('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('FR','QNR')]}
    record['reference'][t][v]={'best_epoch':x['best_epoch'],'checkpoint_sha256':x['best_sha256'],'RR':x['metrics']['RR']['mean'],'FR':x['metrics']['FR']['mean'],'delta_vs_exp1':d}
  (out/'metrics.json').write_text(json.dumps(record,indent=2));report['sensors'][sensor]=record;(a.output/'summary.json').write_text(json.dumps(report,indent=2))
  fig,axes=plt.subplots(1,3,figsize=(13,4));labels=['exp1','original','repeat','F0']
  for ax,(pr,k) in zip(axes,[('RR','PSNR'),('RR','SAM'),('FR','QNR')]):
   vals=[record['protocols'][pr]['mean'][k]]+[reference['150'][v]['metrics'][pr]['mean'][k] for v in labels[1:]];ax.bar(labels,vals);ax.set_title(f'{sensor} {pr} {k}');ax.ticklabel_format(axis='y',style='plain',useOffset=False)
  fig.tight_layout();fig.savefig(out/'comparison.png');plt.close(fig);del net;torch.cuda.empty_cache()
 report['completed_time']=time.time();report['environment']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tf32':False,'precision':'FP32'};(a.output/'summary.json').write_text(json.dumps(report,indent=2));(a.output/'complete.json').write_text(json.dumps({'state':'complete','sensors':list(report['sensors']),'scene_evaluations':40*len(report['sensors']),'time':time.time()},indent=2));print('PASS:',list(report['sensors']),40*len(report['sensors']),'RR/FR scenes',flush=True)
if __name__=='__main__':main()
