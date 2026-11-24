"""One sensor evaluation worker: full RR/FR20, fixed images, no test selection."""
import argparse,importlib.util,json,sys,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'expected_data_manifest.json'
sys.path.insert(0,str(HERE/'SSA-MRN'))
from common.metrics import rr_metrics,fr_metrics
from common import result_images as R,baseline_models as B
from model import Adapter,load
from data import find
from utils import digest,write

def predict(net,s,scale,device):
    args=[torch.from_numpy((s[k]/scale).astype('float32'))[None].to(device) for k in ('pan','lms','ms')]
    if device.startswith('cuda'):torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.no_grad():out=net(*args)
    if device.startswith('cuda'):torch.cuda.synchronize()
    result=out[0].float().cpu().numpy().astype(np.float64)*scale
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite output')
    return result,time.perf_counter()-t

def evaluate(folder,input_root,sensor,device,smoke=False,deadline=float('inf'),stop_file=None):
    folder=Path(folder);scale=1023 if sensor=='GF2' else 2047
    if not (folder/'training_complete.json').exists():raise ValueError('Complete this sensor training before test')
    ck=folder/'best.pt';done=json.loads((folder/'training_complete.json').read_text(encoding='utf8'))
    if digest(ck)!=done['best_sha256']:raise ValueError('Selected checkpoint changed')
    net=Adapter(4,42);load(net,ck);net.to(device).eval();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    baseline=B.build('c2ms_sp',sensor,42,device=device) if not smoke else None
    output=folder/'evaluation';output.mkdir(exist_ok=True)
    if (output/'evaluation_complete.json').exists():return
    names=['c2ms_sp_s42','u2net_retrain_s42'];labels=['C2MS-SP seed42','U2Net retrain seed42']
    from PIL import ImageFont
    font=ImageFont.truetype(str(HERE/'fonts/NanumGothic-Regular.ttf'),15)
    result=dict(sensor=sensor,seed=42,setting='retrain',smoke_excluded=smoke,checkpoint_sha256=digest(ck),registered_parameters=sum(p.numel() for p in net.parameters()),timing_convention='forward plus output CPU transfer, input transfer and metric excluded; first inference includes warmup effects',rr=[],fr=[],data={},metric_note='PSNR is original common metric with peak2047; PSNR_sensor_peak additionally uses actual sensor peak.',images={})
    for split in ('rr','fr'):
        filename='test_'+sensor.lower()+('_OrigScale' if split=='fr' else '')+'_multiExm1.h5';path=find(input_root,filename)
        data_sha=digest(path)
        references=json.loads(REFERENCE.read_text(encoding='utf-8-sig'))['files']
        matching=[r for r in references if r['original_path'].endswith('/'+filename)]
        if len(matching)!=1 or data_sha!=matching[0]['sha256']:raise ValueError('Test data differs from supplied manifest: '+filename)
        result['data'][filename]=dict(sha256=data_sha,bytes=path.stat().st_size)
        with h5py.File(path,'r') as f:
            if len(f['pan'])!=20:raise ValueError('Expected full20 scenes')
            for i in range(1 if smoke else 20):
                if time.time()>=deadline or (stop_file and Path(stop_file).exists()):
                    write(output/'metrics_partial.json',result)
                    write(folder/'status.json',dict(sensor=sensor,phase='evaluation paused',epoch=done['epochs'],epochs=done['epochs']))
                    return 'paused'
                keys=('pan','lms','ms','gt') if split=='rr' else ('pan','lms','ms');s={k:np.asarray(f[k][i],dtype=np.float64) for k in keys}
                pred,seconds=predict(net,s,scale,device)
                m=rr_metrics(s['gt'],pred) if split=='rr' else fr_metrics(pred,s['lms'],s['ms'],s['pan'])
                if split=='rr':m['PSNR_sensor_peak']=m['PSNR']+20*np.log10(scale/2047.)
                result[split].append(dict(scene=i,inference_seconds=seconds,**m));write(output/('metrics_smoke.json' if smoke else 'metrics_partial.json'),result)
                write(folder/'status.json',dict(sensor=sensor,phase='test '+split,epoch=done['epochs'],epochs=done['epochs'],batch=i+1,batches=20))
                if split=='rr' and i in R.SCENES and not smoke:
                    base,_=predict(baseline,s,scale,device);outs=dict(zip(names,(base,pred)))
                    title=f'{sensor} RR scene {i} · seed42 · repository PSNR peak2047; GF2 sensor-peak value in JSON'
                    full=output/f'full_scene_{i:02}.png';zoom=output/f'compare_scene_{i:02}.png'
                    R._full_multi(s,outs,names,title,full,font,labels,[2,1,0]);crop=R._zoom_multi(s,outs,names,title,zoom,font,labels,[2,1,0])
                    bm=rr_metrics(s['gt'],base);bm['PSNR_sensor_peak']=bm['PSNR']+20*np.log10(scale/2047.)
                    result['images'][str(i)]=dict(baseline=bm,candidate=m,zoom=crop,selection='fixed scene; crop favours U2Net')
    for split in ('rr','fr'):
        result[split+'_mean']={k:float(np.mean([x[k] for x in result[split]])) for k in result[split][0] if k not in ('scene','inference_seconds')}
    result['gpu_peak_gib']=torch.cuda.max_memory_allocated()/1024**3 if device.startswith('cuda') else None
    if not smoke:
        ref=json.loads((HERE/'SSA-MRN/docs/images/c2ms_sp/baseline_metrics.json').read_text(encoding='utf8'))['results'][sensor]['c2ms_sp']['per_seed']['42']
        result['vs_c2ms_sp_seed42']={k:result['rr_mean'][k]-ref[k] for k in ('PSNR','SAM','ERGAS','SCC','Q2n')}
        result['baseline_psnr_sensor_peak']=ref['PSNR']+20*np.log10(scale/2047.)
        write(output/'metrics.json',result);write(output/'evaluation_complete.json',dict(metrics_sha256=digest(output/'metrics.json'),best_sha256=digest(ck),rr=20,fr=20,images=12))
    else:write(output/'metrics_smoke.json',result)
    write(folder/'status.json',dict(sensor=sensor,phase='evaluation complete',epoch=done['epochs'],epochs=done['epochs']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--smoke',action='store_true');a=a.parse_args();j=json.loads(a.job.read_text(encoding='utf8'))
    try:evaluate(j['folder'],j['input'],j['sensor'],j['device'],a.smoke,j.get('deadline',float('inf')),j.get('stop_file'))
    except Exception:write(Path(j['folder'])/'error.json',dict(error=traceback.format_exc()));raise
