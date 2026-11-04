"""Evaluate each validation-selected arm against the retained same-seed 53 C2MS (official control), the 57 retrained C2MS
(reference) and the 63 C2MS_L1 (150-epoch L1, reference); summary() then compares the two arms. No tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,ARMS
from prepare import Wald,SENSORS,data_files
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images
LABELS={'C2MS_L1S':'C2MS (L1, 4e-4 단계 감소, 360에폭)','C2MS_L1C':'C2MS (L1, 1e-4 cosine, 360에폭)'}


@torch.inference_mode()
def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    root=HERE.parents[2]
    paths={'C2MS':root/cfg['controls'][sensor],arm:dest/'checkpoints/best.pt','C2MS_57':root/cfg['reference_controls'][sensor],'C2MS_L1_63':root/cfg['controls'][sensor+'_L1_150']}
    models={};weights={}
    for name,path in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed:raise ValueError('control seed mismatch '+name)
        m=Model(arm,SENSORS[sensor]['bands'],seed).cuda();m.load_state_dict(ck['model'],strict=True);models[name]=m.eval()
        weights[name]=dict(path=str(path),sha256=sha(path),best_epoch=ck['best_epoch'])
    rr=Path(cfg['data_root'])/data_files(sensor)['RR'];fr=Path(cfg['data_root'])/data_files(sensor)['FR']
    if sha(rr)!=cfg['data_sha256'][sensor]['RR'] or sha(fr)!=cfg['data_sha256'][sensor]['FR']:raise ValueError('test data mismatch')
    wald=Wald(sensor,'cuda');rows={name:[] for name in models}
    with h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for name,m in models.items():rows[name].append(dict(index=i,metrics=V.scene_metrics(m,a,b,wald,'cuda')))
            print(json.dumps(dict(time_kst=now(),scene=i)),flush=True)
    means={name:{part:{k:float(np.mean([x['metrics'][part][k] for x in values])) for k in values[0]['metrics'][part]} for part in ('RR','RR2','FR_WALD','FR')} for name,values in rows.items()}
    judgement=dict(official_vs_53=V.judge([means[arm]],[means['C2MS']]),reference_vs_57=V.judge([means[arm]],[means['C2MS_57']]),reference_vs_63_l1=V.judge([means[arm]],[means['C2MS_L1_63']]))
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));control_val=torch.load(paths['C2MS'],map_location='cpu',weights_only=False)['metrics']
    best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    validation=dict(arm=best,control_53=control_val,delta_psnr=best['val_psnr']-control_val['val_psnr'],delta_sam=best['val_sam']-control_val['val_sam'])
    gate=judgement['official_vs_53']['label']=='개선' and validation['delta_psnr']>0
    record=dict(sensor=sensor,arm=arm,seed=seed,weights=weights,rows=rows,means=means,judgement=judgement,validation=validation,stage1_gate=gate,selection=cfg['selection'],time_kst=now())
    write(dest/'evaluation'/f'{sensor}_v2_3.json',record)
    result_images.render(models['C2MS'],models[arm],rr.parent,dest/'evaluation'/sensor,names=('C2MS',arm),seed=seed,scenes=cfg['scenes'],
        labels=('현재 기준 C2MS',LABELS[arm]),rr_file=rr.name,scale=SENSORS[sensor]['max_value'],bands=SENSORS[sensor]['rgb'])
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam','lr'),('Train L1 (original scale)','Validation PSNR (peak 1)','Validation SAM (degrees)','Learning rate')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid]);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    axes[1].axhline(control_val['val_psnr'],color='gray',ls='--',lw=1,label='53 C2MS');axes[1].legend(frameon=False)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; best selected by validation MSE');fig.tight_layout();fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,judgement=judgement,validation=validation,stage1_gate=gate,scope='full 20-scene RR/RR2/FR_WALD/FR observation; v2.3; single seed; no automatic baseline promotion'))


def summary(cfg,sensor,seed):
    """Learning-rate effect (L1S against L1C, same loss and length) and length effect (L1C against 63, same loss and rate)."""
    runtime=Path(cfg['runtime']);rec={arm:json.loads((runtime/'runs'/run_name(sensor,arm,seed)/'evaluation'/f'{sensor}_v2_3.json').read_text(encoding='utf8')) for arm in ARMS}
    means={arm:r['means'][arm] for arm,r in rec.items()};l1=rec['C2MS_L1C']['means']['C2MS_L1_63']
    out=dict(time_kst=now(),sensor=sensor,seed=seed,learning_rate_L1S_vs_L1C=V.judge([means['C2MS_L1S']],[means['C2MS_L1C']]),
             length_L1C_vs_63=V.judge([means['C2MS_L1C']],[l1]),official={arm:r['judgement']['official_vs_53'] for arm,r in rec.items()},
             validation={arm:r['validation'] for arm,r in rec.items()},stage1_gate={arm:r['stage1_gate'] for arm,r in rec.items()})
    write(runtime/'comparisons.json',out);print(json.dumps(dict(gate=out['stage1_gate'],lr=out['learning_rate_L1S_vs_L1C']['delta']['RR'],length=out['length_L1C_vs_63']['delta']['RR'])),flush=True)
