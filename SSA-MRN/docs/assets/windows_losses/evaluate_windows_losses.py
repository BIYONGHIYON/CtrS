"""Archive original controlled-suite checkpoints and evaluate frozen QB best models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, datetime, hashlib, json, shutil, sys, time, zipfile
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    a=p.parse_args(); root=a.root; suite=root/'experiments/controlled_suite_v2'; out=a.output
    if out.exists(): raise FileExistsError('Use a fresh evidence output directory')
    out.mkdir(parents=True)
    write(out/'evaluation_state.json', {'status':'running', 'started':datetime.datetime.now().astimezone().isoformat()})
    import torch, h5py, numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    source=json.loads((suite/'source_hashes.json').read_text())
    assert all(sha(root/name)==digest for name,digest in source.items()), 'Training source drift'
    # Snapshot before importing. Nothing in the completed training checkout is edited.
    for name in source:
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/name,target)
    for path in (root/'src').rglob('*.py'):
        target=out/'source'/path.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    upstream=root/'references/upstream/network.py'
    target=out/'source/references/upstream/network.py'; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(upstream,target)
    profile=root/'scripts/observation_profiles/qb.json'
    target=out/'source/scripts/observation_profiles/qb.json'; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(profile,target)
    shutil.copy2(Path(__file__), out/'evaluate_windows_losses.py')
    sys.path.insert(0,str(out/'source/src'))
    from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
    from ssamrn.metrics import rr_metrics, fr_metrics
    for name in ['plan.json','state.json','source_hashes.json','baseline_selection.json','screening_selection.json']:
        shutil.copy2(suite/name,out/name)
    selection=json.loads((suite/'screening_selection.json').read_text()); assert selection['winner']['name']=='04_edge_w0.1_s42'
    names=sorted(d.name for d in suite.iterdir() if d.is_dir() and d.name.startswith(('02_','03_','04_')))+['05_final_candidate']
    assert len(names)==10
    report={'experiment':'windows_losses','export_time':datetime.datetime.now().astimezone().isoformat(),
            'source_root':str(root),'source_hashes_verified':True,'device':torch.cuda.get_device_name(0),
            'torch_version':str(torch.__version__),'selection':'minimum validation MSE; test never used for selection',
            'runs':{},'data':{},'test':{},'limits':['Single seed 42; no repeated-seed significance claim',
            'RR PSNR is per-band mean at peak2047; not verified as paper protocol',
            'FR PAN resize is scikit-image cubic, not verified MATLAB parity; QNR provisional',
            'RR Q2n/SCC not fully MATLAB parity verified',
            'Validation SAM and band-mean PSNR were not logged; not reconstructed',
            'Windows results do not isolate architecture effects against Linux experiments',
            'QB public test previously used in repository research; not a newly untouched holdout']}
    histories={}
    for name in names:
        d=suite/name; complete=json.loads((d/'complete.json').read_text()); config=complete['config']
        rows=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        if name=='05_final_candidate':
            continuation=json.loads((d/'continuation.json').read_text()); assert continuation=={'source':selection['winner']['name'],'resume':'latest','start_epoch':30}
            first=[json.loads(x) for x in (suite/continuation['source']/'history.jsonl').read_text().splitlines()]
            rows=list({x['epoch']:x for x in first+rows}.values()); rows.sort(key=lambda x:x['epoch'])
        target_epochs=100 if name=='05_final_candidate' else 30
        assert complete['epoch']==config['epochs']==target_epochs
        assert [x['epoch'] for x in rows]==list(range(1,target_epochs+1))
        best=min(rows,key=lambda x:x['val_mse']); assert best['val_mse']==complete['best_mse']
        item={'source':str(d),'config':config,'epochs':target_epochs,'best_epoch':best['epoch'],
              'best_validation_mse':best['val_mse'],'checkpoints':{}}
        for kind in ['best','latest']:
            path=d/(kind+'.pt'); ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['config']==config
            assert ck['epoch']==(best['epoch'] if kind=='best' else target_epochs)
            assert ck['best_mse']==best['val_mse']
            assert all(k in ck for k in ['model','optimizer','rng','cuda_rng','loader_rng'])
            item['checkpoints'][kind]={'epoch':ck['epoch'],'bytes':path.stat().st_size,'sha256':sha(path),
                                     'format':'original model/config/Adam/CPU-CUDA-loader RNG checkpoint'}
        for filename in ['best.pt','latest.pt','config.json','complete.json','history.jsonl']+(['continuation.json'] if name=='05_final_candidate' else []):
            dest=out/'runs'/name/filename; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(d/filename,dest)
            assert sha(dest)==sha(d/filename)
        if name=='05_final_candidate': write(out/'final_combined_history.json',rows)
        histories[name]=rows; report['runs'][name]=item
    base=suite/'01_QB_k4_s42'
    base_history=[json.loads(x) for x in (base/'history.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in base_history]==list(range(1,101))
    write(out/'baseline_reference.json',{'source':str(base),'history':base_history,
          'best_validation_mse':min(x['val_mse'] for x in base_history),
          'checkpoint_sha256':sha(base/'best.pt'),
          'git_archive':'../windows_k_baselines/runs/01_QB_k4_s42/best.pt'})
    paths={'RR':a.data_root/'QuickBird/Testing Dataset (ReducedData, H5 Format)/test_qb_multiExm1.h5',
           'FR':a.data_root/'QuickBird/Testing Dataset (FullData, H5 Format)/test_qb_OrigScale_multiExm1.h5'}
    with h5py.File(paths['RR']) as f: count=len(f['pan'])
    scenes={'seed':42,'protocol':'RR','zero_based_scene_indices':sorted(np.random.default_rng(42).choice(count,5,replace=False).tolist()),
            'rule':'frozen before reading test predictions/metrics; same scenes for baseline and candidate',
            'scene_id':'full H5 example index; original acquisition IDs unavailable'}
    write(out/'scene_selection.json',scenes)
    for protocol,path in paths.items():
        with h5py.File(path) as f: shapes={k:list(v.shape) for k,v in f.items()}
        report['data'][protocol]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'shapes':shapes}
    def panels(arr,pred,path,label,index):
        bands=[2,1,0]; gt=arr['gt'][bands].transpose(1,2,0); lo=np.percentile(gt,1,axis=(0,1)); hi=np.percentile(gt,99,axis=(0,1))
        def rgb(x):return Image.fromarray((255*np.clip((x[bands].transpose(1,2,0)-lo)/np.maximum(hi-lo,1e-8),0,1)).astype('uint8'))
        pan=arr['pan'][0]; p0,p1=np.percentile(pan,[1,99]); gray=Image.fromarray((255*np.clip((pan-p0)/max(p1-p0,1e-8),0,1)).astype('uint8')).convert('RGB')
        h,w=pan.shape; canvas=Image.new('RGB',(w*4,h+38),'white'); draw=ImageDraw.Draw(canvas)
        for j,(title,im) in enumerate([('LR MS',rgb(arr['ms'])),('PAN',gray),(label,rgb(pred)),('GT',rgb(arr['gt']))]):
            canvas.paste(im.resize((w,h),Image.Resampling.NEAREST),(j*w,38)); draw.text((j*w+8,6),title,fill='black')
        draw.text((8,23),f'QB RR index {index} | full scene | shared GT 1-99% stretch',fill='black'); canvas.save(path)
        return {'bands_zero_based':bands,'gt_percentile_lo':lo.tolist(),'gt_percentile_hi':hi.tolist(),'pan_lo_hi':[float(p0),float(p1)]}
    for label,d in [('baseline',base),('edge_w0.1',suite/'05_final_candidate')]:
        ck=torch.load(d/'best.pt',map_location='cpu',weights_only=True); c=ck['config']; assert c['sensor']=='QB' and c['k']==4
        with h5py.File(paths['RR']) as f: channels=f['lms'].shape[1]
        model=RestoredPansharpeningNet(channels=channels,ssai_dimension=c['k']); model.load_state_dict(ck['model'],strict=True); model=model.cuda().eval()
        result={'checkpoint':str(d/'best.pt'),'checkpoint_sha256':sha(d/'best.pt'),'checkpoint_epoch':ck['epoch'],'metrics':{},'images':[]}
        for protocol,path in paths.items():
            rows=[]; started=time.monotonic()
            with h5py.File(path) as f:
                required={'pan','lms','ms'}|({'gt'} if protocol=='RR' else set()); assert set(f.keys())==required
                for index in range(len(f['pan'])):
                    arr={k:np.asarray(f[k][index],dtype=np.float32) for k in required}; assert all(np.isfinite(v).all() for v in arr.values())
                    inputs=[torch.from_numpy(arr[k]/2047.).unsqueeze(0).cuda() for k in ['pan','lms','ms']]
                    with torch.inference_mode(): pred=model(*inputs)[0].cpu().numpy().astype(np.float64)*2047.
                    assert np.isfinite(pred).all()
                    metrics=rr_metrics(arr['gt'],pred) if protocol=='RR' else fr_metrics(pred,arr['lms'],arr['ms'],arr['pan'])
                    if protocol=='RR':
                        metrics['MSE_peak1']=float(np.mean(((pred-arr['gt'])/2047.)**2))
                        if index in scenes['zero_based_scene_indices']:
                            name=f'{label}_scene_{index:02}.png'; display=panels(arr,pred,out/name,label,index); result['images'].append({'index':index,'file':name,'display':display})
                    assert all(np.isfinite(v) for v in metrics.values()); rows.append({'index':index,'metrics':metrics})
                    print(label,protocol,index+1,len(f['pan']),flush=True)
            result['metrics'][protocol]={'count':len(rows),'mean':{k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']},'rows':rows,'seconds':time.monotonic()-started}
        report['test'][label]=result; write(out/'metrics.json',report)
        del model,ck; torch.cuda.empty_cache()
    report['delta_from_baseline']={p:{k:v-report['test']['baseline']['metrics'][p]['mean'][k] for k,v in report['test']['edge_w0.1']['metrics'][p]['mean'].items()} for p in ['RR','FR']}
    write(out/'metrics.json',report)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,h in [('K4 baseline',base_history),('K4 + edge 0.1',histories['05_final_candidate'])]:
        for ax,key in zip(axes[:2],['train_mse','val_mse']):ax.plot([x['epoch'] for x in h],[x[key] for x in h],label=name)
        axes[2].plot([x['epoch'] for x in h],[-10*np.log10(x['val_mse']) for x in h],label=name)
    for ax,title in zip(axes,['Train MSE','Validation MSE','Derived global validation PSNR (peak1)']):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_yscale('log');axes[1].set_yscale('log');fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4)); labels=[n.replace('02_','').replace('03_','').replace('04_','').replace('_s42','') for n in names[:-1]]
    ax.bar(labels,[report['runs'][n]['best_validation_mse'] for n in names[:-1]],color=['#4c78a8']*3+['#59a14f']*3+['#f28e2b']*3)
    ax.axhline(min(x['val_mse'] for x in base_history[:30]),color='black',linestyle='--',label='baseline best within 30 epochs')
    ax.set_ylabel('Best validation MSE (30 epochs)');ax.tick_params(axis='x',rotation=25);ax.legend();fig.tight_layout();fig.savefig(out/'screening.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5))
    for ax,(protocol,key) in zip(axes,[('RR','PSNR'),('RR','SAM'),('RR','ERGAS'),('FR','QNR')]):
        delta=report['delta_from_baseline'][protocol][key];ax.bar(['edge 0.1'],[delta]);ax.axhline(0,color='black',linewidth=.7);ax.set_title(f'{protocol} {key} delta');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same-server best checkpoints: candidate minus baseline (FR provisional)');fig.tight_layout();fig.savefig(out/'test_deltas.png',dpi=150);plt.close(fig)
    assert all(sha(root/name)==digest for name,digest in source.items())
    write(out/'evaluation_state.json',{'status':'finished','finished':datetime.datetime.now().astimezone().isoformat(),'training_epochs':940})
    artifacts={str(path.relative_to(out)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(out.rglob('*')) if path.is_file()}
    write(out/'report_manifest.json',{'artifacts':artifacts,'original_checkpoint_count':20,'source_hashes_verified':True})
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(out)))
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'status':'finished'}),flush=True)
if __name__=='__main__': main()
