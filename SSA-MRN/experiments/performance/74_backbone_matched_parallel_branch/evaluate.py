"""Evaluate the validation-selected C2MS_M against the retained same-seed 53 C2MS (official control, criteria v2.4: RR only),
the 57 retrained C2MS (run-to-run reference) and the original SSA-MRN 43 C0 (reference for the share of the C2MS gain kept).
RR2 and FR_WALD come out of the shared scene_metrics but are dropped from the record; FR QNR/D_lambda/D_s are kept as reference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,M43
from prepare import Wald,SENSORS,data_files
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images
LABEL='C2MS (W2 가지를 원본 backbone과 같은 폭·깊이로)'
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4, single seed


def judge(cand,ctrl,seeds=1):
    """Criteria v2.4: RR PSNR, SAM and ERGAS all better and the PSNR gain at or above the noise threshold."""
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,
                noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


@torch.inference_mode()
def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    root=HERE.parents[2];import model as M
    paths={'C2MS':(root/cfg['controls'][sensor],'C2MS'),arm:(dest/'checkpoints/best.pt',arm),'C2MS_57':(root/cfg['reference_controls'][sensor],'C2MS'),'C0':(root/cfg['original'][sensor],'C0')}
    models={};weights={}
    for name,(path,kind) in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed:raise ValueError('seed mismatch '+name)
        m=M43.Model('C0',seed) if kind=='C0' else M.M54.Model('C2MS',SENSORS[sensor]['bands'],seed) if kind=='C2MS' else Model(arm,SENSORS[sensor]['bands'],seed)
        m.load_state_dict(ck['model'],strict=True);models[name]=m.cuda().eval();weights[name]=dict(path=str(path),sha256=sha(path),best_epoch=ck['best_epoch'])
    rr=Path(cfg['data_root'])/data_files(sensor)['RR'];fr=Path(cfg['data_root'])/data_files(sensor)['FR']
    if sha(rr)!=cfg['data_sha256'][sensor]['RR'] or sha(fr)!=cfg['data_sha256'][sensor]['FR']:raise ValueError('test data mismatch')
    wald=Wald(sensor,'cuda');rows={name:[] for name in models}
    with h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for name,m in models.items():
                s=V.scene_metrics(m,a,b,wald,'cuda');rows[name].append(dict(index=i,metrics={'RR':s['RR'],'FR':{k:v for k,v in s['FR'].items() if k!='OBS_RMSE_DN'}}))
            print(json.dumps(dict(time_kst=now(),scene=i)),flush=True)
    means={name:{part:{k:float(np.mean([x['metrics'][part][k] for x in values])) for k in values[0]['metrics'][part]} for part in ('RR','FR')} for name,values in rows.items()}
    judgement=dict(official_vs_53=judge(means[arm],means['C2MS']),reference_vs_57=judge(means[arm],means['C2MS_57']),reference_vs_original=judge(means[arm],means['C0']))
    gain=means['C2MS']['RR']['PSNR']-means['C0']['RR']['PSNR']
    retention=dict(c2ms_gain_dB=gain,arm_gain_dB=means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'],share=(means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain)
    wins={name:dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[arm],rows[name])),
                    SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[arm],rows[name]))) for name in ('C2MS','C2MS_57','C0')}
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));control_val=torch.load(paths['C2MS'][0],map_location='cpu',weights_only=False)['metrics']
    best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    validation=dict(arm=best,control_53=control_val,delta_psnr=best['val_psnr']-control_val['val_psnr'],delta_sam=best['val_sam']-control_val['val_sam'])
    parameters={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()}
    record=dict(sensor=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)',weights=weights,parameters=parameters,rows=rows,means=means,judgement=judgement,
                retention=retention,scene_wins=wins,validation=validation,selection=cfg['selection'],time_kst=now())
    write(dest/'evaluation'/f'{sensor}_v2_4.json',record)
    result_images.render(models['C2MS'],models[arm],rr.parent,dest/'evaluation'/sensor,names=('C2MS',arm),seed=seed,scenes=cfg['scenes'],
        labels=('현재 기준 C2MS',LABEL),rr_file=rr.name,scale=SENSORS[sensor]['max_value'],bands=SENSORS[sensor]['rgb'])
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam','lr'),('Train MSE (original scale)','Validation PSNR (peak 1)','Validation SAM (degrees)','Learning rate')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid]);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    axes[1].axhline(control_val['val_psnr'],color='gray',ls='--',lw=1,label='53 C2MS');axes[1].legend(frameon=False)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; best selected by validation MSE');fig.tight_layout();fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,key in zip(axes,('PSNR','SAM','ERGAS')):
        names=('C0','C2MS_57',arm);vals=[means[n]['RR'][key]-means['C2MS']['RR'][key] for n in names]
        ax.bar(['original C0','57 C2MS',arm],vals,color=['#8d99ae','#adb5bd','#d1495b']);ax.axhline(0,color='black',lw=.8);ax.set(title=f'QB test RR {key} - 53 C2MS');ax.grid(axis='y',alpha=.3)
    fig.tight_layout();fig.savefig(dest/'evaluation/comparison.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,parameters=parameters,judgement=judgement,retention=retention,validation=validation,
                    scope='full 20-scene RR (judgement) and FR QNR (reference); v2.4; single seed; no automatic baseline promotion'))
