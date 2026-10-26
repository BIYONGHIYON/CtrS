"""Evaluate the validation-selected C2MS trained with L1 + spectral loss against the retained same-seed 53 C2MS (official control) and the references in config reference_controls; no tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model
from prepare import Wald,SENSORS,data_files
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images

@torch.inference_mode()
def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed)
    done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    paths={'C2MS':HERE.parents[2]/cfg['controls'][sensor],'C2MS_L1S':dest/'checkpoints/best.pt'}
    refs=cfg.get('reference_controls',{}).get(sensor,{})
    paths.update({name:HERE.parents[2]/p for name,p in refs.items()})
    models={};weights={}
    for name,path in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed:raise ValueError('control seed mismatch')
        m=Model('C2MS',SENSORS[sensor]['bands'],seed).cuda();m.load_state_dict(ck['model'],strict=True);models[name]=m.eval()
        weights[name]=dict(path=str(path),sha256=sha(path),best_epoch=ck['best_epoch'])
    summary={}
    for target in [sensor]+(['WV2'] if sensor=='WV3' else []):
        rr=Path(cfg['data_root'])/data_files(target)['RR'];fr=Path(cfg['data_root'])/data_files(target)['FR']
        expected=(cfg['data_sha256'] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'])[target]
        if sha(rr)!=expected['RR'] or sha(fr)!=expected['FR']:raise ValueError('test data mismatch')
        wald=Wald(target,'cuda');rows={name:[] for name in models}
        with h5py.File(rr) as r,h5py.File(fr) as f:
            if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
            for i in range(20):
                a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')}
                b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
                for name,m in models.items():rows[name].append(dict(index=i,metrics=V.scene_metrics(m,a,b,wald,'cuda')))
                print(json.dumps(dict(time_kst=now(),target=target,scene=i)),flush=True)
        means={name:{part:{k:float(np.mean([x['metrics'][part][k] for x in values])) for k in values[0]['metrics'][part]} for part in ('RR','RR2','FR_WALD','FR')} for name,values in rows.items()}
        judgement=V.judge([means['C2MS_L1S']],[means['C2MS']])
        reference={name:V.judge([means['C2MS_L1S']],[means[name]]) for name in refs}
        record=dict(target=target,training_sensor=sensor,seed=seed,weights=weights,rows=rows,means=means,judgement=judgement,judgement_vs_references=reference,selection=cfg['selection'],time_kst=now())
        write(dest/'evaluation'/f'{target}_v2_3.json',record);summary[target]=dict(means=means,judgement=judgement,judgement_vs_references=reference)
        result_images.render(models['C2MS'],models['C2MS_L1S'],rr.parent,dest/'evaluation'/target,names=('C2MS','C2MS-L1S'),seed=seed,scenes=cfg['scenes'],
            labels=('현재 기준 C2MS','C2MS (L1 + 분광 손실)'),rr_file=rr.name,scale=SENSORS[target]['max_value'],bands=SENSORS[target]['rgb'],
            land_cover=result_images.LAND_COVER if target=='QB' else {i:f'{target} 장면 {i}' for i in cfg['scenes']})
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    h=json.loads((dest/'history.json').read_text(encoding='utf8'));fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam'),('Train L1 (original scale)','Validation PSNR (peak 1)','Validation SAM (degrees)')):
        valid=[x for x in h if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid]);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    fig.suptitle(f'{sensor} C2MS trained with L1 + spectral loss, seed {seed}; best selected by validation MSE');fig.tight_layout();fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    targets=list(summary);fig,axes=plt.subplots(1,len(targets),figsize=(5*len(targets),3.5),squeeze=False)
    for ax,target in zip(axes[0],targets):
        parts=('RR','RR2','FR_WALD');delta=summary[target]['judgement']['delta'];ax.bar(parts,[delta[p]['PSNR'] for p in parts]);ax.axhline(0,color='black',lw=.8);ax.set(title=target,ylabel='PSNR delta: C2MS-L1S - C2MS (dB)')
    fig.tight_layout();fig.savefig(dest/'evaluation/comparison.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,targets=summary,scope='full 20-scene RR/RR2/FR_WALD/FR observation; v2.3; single seed; no automatic baseline promotion'))
