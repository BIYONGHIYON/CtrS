"""Evaluate frozen best QB checkpoints on complete RR/FR sets; export evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
from pathlib import Path
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ssamrn.models.pan_variants import make_model
from ssamrn.metrics import rr_metrics,fr_metrics
VARIANTS=['baseline','interp23','lr_correction','high_frequency']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


def panels(sample,prediction,path,label,index):
    bands=[2,1,0]
    reference=sample['gt'][bands].transpose(1,2,0)
    lo=np.percentile(reference,1,axis=(0,1));hi=np.percentile(reference,99,axis=(0,1))
    def rgb(a):
        a=a[bands].transpose(1,2,0)
        return Image.fromarray((np.clip((a-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
    pan=sample['pan'][0];p0,p1=np.percentile(pan,[1,99])
    gray=Image.fromarray((np.clip((pan-p0)/max(p1-p0,1e-8),0,1)*255).astype('uint8')).convert('RGB')
    images=[('LR MS',rgb(sample['ms'])),('PAN',gray),(label,rgb(prediction)),('GT',rgb(sample['gt']))]
    h,w=pan.shape;canvas=Image.new('RGB',(4*w,h+38),'white');draw=ImageDraw.Draw(canvas)
    for j,(name,image) in enumerate(images):
        canvas.paste(image.resize((w,h),Image.Resampling.NEAREST),(j*w,38));draw.text((j*w+8,6),name,fill='black')
    draw.text((8,23),f'QB RR index {index} | full scene | shared GT 1-99% display stretch',fill='black')
    canvas.save(path)
    return {'rgb_assumed_bands_zero_based':bands,'display_lo':lo.tolist(),'display_hi':hi.tolist(),'pan_display_lo_hi':[float(p0),float(p1)],'no_inference_crop':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/a6000_architecture_qb');a=parser.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    run=ROOT/'experiments/a6000_architecture_qb';data=ROOT/'data/dataset/QuickBird'
    paths={'RR':data/'Testing Dataset (ReducedData, H5 Format)/test_qb_multiExm1.h5','FR':data/'Testing Dataset (FullData, H5 Format)/test_qb_OrigScale_multiExm1.h5'}
    # Freeze before loading checkpoints or computing any test scores.
    with h5py.File(paths['RR'],'r') as f:count=len(f['pan'])
    selection={'seed':42,'protocol':'RR','zero_based_scene_indices':sorted(np.random.default_rng(42).choice(count,5,replace=False).tolist()),'tiles':'full scene','rule':'fixed before test evaluation, not chosen by metric'}
    selection_file=out/'selection.json'
    if selection_file.exists():assert json.loads(selection_file.read_text())==selection
    else:write(selection_file,selection)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    report={'sensor':'QB','device':'cuda','checkpoint_selection':'best validation MSE; test never used for selection','selection':selection,'data':{},'variants':{},'limits':['FR PAN resize uses scikit-image, not MATLAB parity verified','Q4/SCC implementation not fully MATLAB parity verified','PSNR uses per-band peak2047 arithmetic mean','Validation SAM was not logged; no SAM curve reconstructed','Single seed; no repeated-seed improvement claim']}
    for protocol,path in paths.items():
        with h5py.File(path,'r') as f:shapes={k:list(v.shape) for k,v in f.items()}
        report['data'][protocol]={'file':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'shapes':shapes}
    for variant in VARIANTS:
        d=run/f'QB_{variant}_k6_s42';record={'history':[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()],'checkpoints':{}}
        (out/'weights').mkdir(exist_ok=True)
        for kind in ['best','latest']:
            path=d/(kind+'.pt');ck=torch.load(path,map_location='cpu',weights_only=True)
            artifact=out/'weights'/f'{variant}_{kind}.pt'
            # Export real learned inference weights; full Adam/RNG resumes remain on server.
            torch.save({k:ck[k] for k in ['model','config','epoch','best_mse']},artifact)
            record['checkpoints'][kind]={'epoch':ck['epoch'],'full_server_path':str(path),'full_sha256':sha(path),'inference_file':str(artifact.relative_to(out)),'inference_sha256':sha(artifact)}
            if kind=='best':best=ck
        model=make_model(best['config']);model.load_state_dict(best['model'],strict=True);model=model.cuda().eval()
        record['config']=best['config'];record['metrics']={};record['images']=[]
        for protocol,path in paths.items():
            rows=[];started=time.monotonic()
            with h5py.File(path,'r') as f:
                for index in range(len(f['pan'])):
                    arrays={k:np.asarray(v[index],dtype=np.float32) for k,v in f.items()}
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    inputs=[torch.from_numpy(arrays[k]/2047.).unsqueeze(0).cuda() for k in ['pan','lms','ms']]
                    with torch.inference_mode():prediction=model(*inputs)[0].cpu().numpy().astype(np.float64)*2047.
                    assert np.isfinite(prediction).all()
                    if protocol=='RR':
                        metrics=rr_metrics(arrays['gt'],prediction);metrics['MSE_peak1']=float(np.mean(((prediction-arrays['gt'])/2047.)**2))
                        if variant!='baseline' and index in selection['zero_based_scene_indices']:
                            name=f'{variant}_scene_{index:02}.png';display=panels(arrays,prediction,out/name,variant,index)
                            record['images'].append(dict(index=index,file=name,display=display))
                    else:metrics=fr_metrics(prediction,arrays['lms'],arrays['ms'],arrays['pan'])
                    rows.append(dict(index=index,metrics=metrics))
            mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
            record['metrics'][protocol]={'count':len(rows),'mean':mean,'rows':rows,'seconds':time.monotonic()-started}
            print(variant,protocol,json.dumps(mean),flush=True)
        report['variants'][variant]=record
        del model,best;torch.cuda.empty_cache()
        write(out/'metrics.json',report)
    base=report['variants']['baseline']['metrics']
    for variant,record in report['variants'].items():
        record['delta_from_baseline']={p:{k:v-base[p]['mean'][k] for k,v in record['metrics'][p]['mean'].items()} for p in ['RR','FR']}
    write(out/'metrics.json',report)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for variant,r in report['variants'].items():
        h=r['history'];epochs=[x['epoch'] for x in h]
        color={'baseline':'#777777','interp23':'#2878b5','lr_correction':'#f39c34','high_frequency':'#3f9c6c'}[variant]
        axes[0].plot(epochs,[x['train_mse'] for x in h],label=variant,color=color)
        axes[1].plot(epochs,[x['val_mse'] for x in h],label=variant,color=color)
        axes[2].plot(epochs,[-10*np.log10(x['val_mse']) for x in h],label=variant,color=color)
    for ax,title in zip(axes,['Train MSE','Validation MSE','Validation global PSNR (derived, peak1)']):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    axes[0].set_yscale('log');axes[1].set_yscale('log');axes[2].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'learning.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    colors=['#2878b5','#f39c34','#3f9c6c']
    for ax,(protocol,key,title) in zip(axes.flat,[('RR','PSNR','RR PSNR delta (up is better)'),('RR','SAM','RR SAM delta (down is better)'),('RR','ERGAS','RR ERGAS delta (down is better)'),('RR','Q2n','RR Q4 delta (up is better)'),('FR','QNR','FR QNR delta (provisional; up is better)'),('RR','MSE_peak1','RR MSE delta (down is better)')]):
        values=[report['variants'][v]['delta_from_baseline'][protocol][key] for v in VARIANTS[1:]]
        ax.bar(range(3),values,color=colors);ax.set_xticks(range(3),['23tap','LR correction','high frequency'],rotation=15,fontsize=8);ax.set_title(title);ax.axhline(0,color='black',linewidth=.8);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Difference from the same-server baseline (variant minus baseline)')
    fig.tight_layout();fig.savefig(out/'test_metrics.png',dpi=160);plt.close(fig)
    artifacts={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report_manifest.json'}
    write(out/'report_manifest.json',{'artifacts':artifacts,'code_evaluator_sha256':sha(Path(__file__)),'best_selection':'validation only','weights_format':'inference config/model/epoch; full resume checkpoints remain on server'})
    print('COMPLETE',len(artifacts),'artifacts',flush=True)


if __name__=='__main__':main()
