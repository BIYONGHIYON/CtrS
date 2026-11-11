"""Evaluate each validation-selected student (from 77 evaluate.py).

Models on the same 20 QB RR/FR test scenes and the same code:
- 53 C2MS, same seed (official control, criteria v2.4: RR only);
- the same structure trained on GT only (77 E3 / 75 W64D8, seed 42): the stage 1 gate (does the teacher term help this structure);
- 57 retrained C2MS (run-to-run reference), the original SSA-MRN 43 C0 (share of the C2MS gain kept) and the teacher E_ALL (reference).
RR2 and FR_WALD come out of the shared scene_metrics but are dropped; FR QNR/D_lambda/D_s are kept as reference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,Teacher,M43,M54,conv_macs
from prepare import Wald,SENSORS,data_files,teacher_paths
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images
LABELS={'C2MS_B64D8_W64D8_KD':'E3 구조 + 앙상블 교사 증류','C2MS_W64D8_KD':'W64D8 구조 + 앙상블 교사 증류'}
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
    root=HERE.parents[2];bands=SENSORS[sensor]['bands']
    same=cfg['same_structure_controls'][arm]
    paths={'C2MS':(root/cfg['controls'][sensor].replace('<seed>',str(seed)),'C2MS'),arm:(dest/'checkpoints/best.pt',arm),
           'C2MS_57':(root/cfg['reference_controls'][sensor],'C2MS'),'C0':(root/cfg['original'][sensor],'C0')}
    if seed==same['seed']:paths['SAME']=(root/same['path'],arm)
    models={};weights={}
    for name,(path,kind) in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if name not in ('C2MS_57','C0') and ck['seed']!=seed:raise ValueError('seed mismatch '+name)
        if name in ('C2MS_57','C0') and ck['seed']!=42:raise ValueError('reference seed mismatch '+name)
        m=M43.Model('C0',ck['seed']) if kind=='C0' else M54.Model('C2MS',bands,ck['seed']) if kind=='C2MS' else Model(arm,bands,seed)
        m.load_state_dict(ck['model'],strict=True);models[name]=m.cuda().eval();weights[name]=dict(path=str(path),sha256=sha(path),best_epoch=ck['best_epoch'])
    models['TEACHER']=Teacher(teacher_paths(cfg),bands).cuda().eval();weights['TEACHER']=dict(paths=[str(p) for p in teacher_paths(cfg)],sha256=[sha(p) for p in teacher_paths(cfg)])
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
    judgement=dict(official_vs_53=judge(means[arm],means['C2MS']),reference_vs_57=judge(means[arm],means['C2MS_57']),
                   reference_vs_original=judge(means[arm],means['C0']),reference_teacher_vs_53=judge(means['TEACHER'],means['C2MS']))
    if 'SAME' in models:judgement['stage1_gate_vs_same_structure']=judge(means[arm],means['SAME'])
    gain=means['C2MS']['RR']['PSNR']-means['C0']['RR']['PSNR']
    retention=dict(c2ms_gain_dB=gain,arm_gain_dB=means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'],share=(means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain)
    if 'SAME' in models:retention['same_structure_share']=(means['SAME']['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain
    others=[n for n in ('C2MS','SAME','C2MS_57','C0','TEACHER') if n in models]
    wins={name:dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[arm],rows[name])),
                    SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[arm],rows[name]))) for name in others}
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    control_val=torch.load(paths['C2MS'][0],map_location='cpu',weights_only=False)['metrics']
    validation=dict(arm=best,control_53=control_val,delta_psnr_vs_53=best['val_psnr']-control_val['val_psnr'],delta_sam_vs_53=best['val_sam']-control_val['val_sam'])
    if 'SAME' in models:
        same_val=torch.load(paths['SAME'][0],map_location='cpu',weights_only=False)['metrics']
        validation.update(same_structure=same_val,delta_psnr_vs_same=best['val_psnr']-same_val['val_psnr'],delta_sam_vs_same=best['val_sam']-same_val['val_sam'])
    shapes=[(1,1,256,256),(1,bands,256,256),(1,bands,64,64)]
    cost={name:dict(parameters=sum(p.numel() for p in m.parameters()),conv_gmac_256=conv_macs(m,shapes)/1e9) for name,m in models.items() if name!='TEACHER'}
    record=dict(sensor=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)',weights=weights,cost=cost,rows=rows,means=means,judgement=judgement,
                retention=retention,scene_wins=wins,validation=validation,selection=cfg['selection'],time_kst=now())
    write(dest/'evaluation'/f'{sensor}_v2_4.json',record)
    result_images.render(models['C2MS'],models[arm],rr.parent,dest/'evaluation'/sensor,names=('C2MS',arm),seed=seed,scenes=cfg['scenes'],
        labels=('현재 기준 C2MS',LABELS[arm]),rr_file=rr.name,scale=SENSORS[sensor]['max_value'],bands=SENSORS[sensor]['rgb'])
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(21,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam','val_teacher_mse','lr'),('Train MSE vs GT (original scale)','Validation PSNR (peak 1)','Validation SAM (degrees)','Validation MSE vs teacher','Learning rate')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid]);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    axes[1].axhline(control_val['val_psnr'],color='gray',ls='--',lw=1,label='53 C2MS')
    if 'SAME' in models:axes[1].axhline(validation['same_structure']['val_psnr'],color='#d1495b',ls=':',lw=1,label='same structure, GT only')
    axes[1].legend(frameon=False);axes[3].set_yscale('log')
    fig.suptitle(f'{sensor} {arm}, seed {seed}; best selected by validation MSE against GT');fig.tight_layout();fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3.6))
    names=[n for n in ('C0','SAME','C2MS_57',arm,'TEACHER') if n in models];labels={'C0':'original C0','SAME':'same structure\nGT only','C2MS_57':'57 C2MS',arm:'student','TEACHER':'teacher E_ALL'}
    colors={'C0':'#8d99ae','SAME':'#f4a261','C2MS_57':'#adb5bd',arm:'#d1495b','TEACHER':'#2a9d8f'}
    for ax,key in zip(axes,('PSNR','SAM','ERGAS')):
        ax.bar([labels[n] for n in names],[means[n]['RR'][key]-means['C2MS']['RR'][key] for n in names],color=[colors[n] for n in names]);ax.axhline(0,color='black',lw=.8)
        ax.set(title=f'{sensor} test RR {key} - 53 C2MS');ax.grid(axis='y',alpha=.3);ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(dest/'evaluation/comparison.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,cost=cost,judgement=judgement,retention=retention,validation=validation,
                    scope='full 20-scene RR (judgement) and FR QNR (reference); v2.4; single seed; no automatic baseline promotion'))
