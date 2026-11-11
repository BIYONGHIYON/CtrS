"""Evaluate each validation-selected arm (from 77/78 evaluate.py) on the 20 QB RR/FR test scenes with the same code for every model.

Systems (same seed): the arm, 53 C2MS (official control), 65 C2MS_L1S (same loss and data, the C2MS branch: the structure effect) and the
original SSA-MRN 43 C0 (share of the C2MS gain kept); the arm, C2MS and C2MS_L1S are also evaluated with the inference projection of
45/66 (`+P`, prepare.Projected) applied to every inference. The official judgement (criteria v2.4, RR only) compares the arm with the
projection against plain C2MS; the arm against C2MS + P and the arm against C2MS_L1S are reported beside it. RR2/FR_WALD come out of the
shared scene_metrics but are dropped; FR QNR/D_lambda/D_s are kept as reference. Seed 42 (stage 1, no other GPU job running) also
records the GPU inference time of C2MS and the arm, with and without the projection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,M43,M54,SHAPES,conv_macs,used_parameters
from prepare import Wald,SENSORS,data_files,Projected
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images
LABELS={'C2MS_W64D12_L1S':'W2 64×12 + L1·분광 + 사영','C2MS_W64D16_L1S':'W2 64×16 + L1·분광 + 사영'}
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4: single seed 0.06 dB, two or more seeds 0.04 dB


def judge(cand,ctrl,seeds=1):
    """Criteria v2.4: RR PSNR, SAM and ERGAS all better and the PSNR gain at or above the noise threshold."""
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,
                noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def comparisons(arm):
    """(name, candidate, control): the official one first."""
    return [('official_arm_P_vs_53',arm+'+P','C2MS'),('arm_vs_53',arm,'C2MS'),('arm_P_vs_53_P',arm+'+P','C2MS+P'),
            ('structure_arm_vs_65',arm,'L1S'),('structure_arm_P_vs_65_P',arm+'+P','L1S+P'),('reference_65_P_vs_53',  'L1S+P','C2MS'),
            ('reference_53_P_vs_53','C2MS+P','C2MS'),('reference_arm_vs_original',arm,'C0')]


def latency(net,x,repeats):
    """Median seconds per forward (CUDA synchronized, 3 warm-up passes); inputs already on the GPU."""
    with torch.inference_mode():
        for _ in range(3):net(*x)
        torch.cuda.synchronize();times=[]
        for _ in range(repeats):
            t=time.perf_counter();net(*x);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    return float(np.median(times))


@torch.inference_mode()
def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    root=HERE.parents[2];bands=SENSORS[sensor]['bands'];seeded=lambda p:root/p.replace('<seed>',str(seed))
    paths={'C2MS':(seeded(cfg['controls'][sensor]),'C2MS'),'L1S':(seeded(cfg['same_loss_controls'][sensor]),'C2MS'),
           arm:(dest/'checkpoints/best.pt',arm),'C0':(seeded(cfg['original'][sensor]),'C0')}
    models={};weights={}
    for name,(path,kind) in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed:raise ValueError('seed mismatch '+name)
        m=M43.Model('C0',seed) if kind=='C0' else M54.Model('C2MS',bands,seed) if kind=='C2MS' else Model(arm,bands,seed)
        m.load_state_dict(ck['model'],strict=True);models[name]=m.cuda().eval();weights[name]=dict(path=str(path),sha256=sha(path),best_epoch=ck['best_epoch'])
    for name in ('C2MS','L1S',arm):models[name+'+P']=Projected(models[name],sensor)
    rr=Path(cfg['data_root'])/data_files(sensor)['RR'];fr=Path(cfg['data_root'])/data_files(sensor)['FR']
    if sha(rr)!=cfg['data_sha256'][sensor]['RR'] or sha(fr)!=cfg['data_sha256'][sensor]['FR']:raise ValueError('test data mismatch')
    wald=Wald(sensor,'cuda');rows={name:[] for name in models}
    with h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for name,m in models.items():
                s=V.scene_metrics(m,a,b,wald,'cuda');rows[name].append(dict(index=i,metrics={'RR':s['RR'],'FR':{k:v for k,v in s['FR'].items() if k!='OBS_RMSE_DN'},'OBS_RMSE_DN':s['FR']['OBS_RMSE_DN']}))
            print(json.dumps(dict(time_kst=now(),scene=i)),flush=True)
    means={name:{part:{k:float(np.mean([x['metrics'][part][k] for x in values])) for k in values[0]['metrics'][part]} for part in ('RR','FR')} for name,values in rows.items()}
    obs={name:float(np.mean([x['metrics']['OBS_RMSE_DN'] for x in values])) for name,values in rows.items()}
    iterations={name:dict(mean=float(np.mean(m.iterations)),max=int(max(m.iterations)),calls=len(m.iterations)) for name,m in models.items() if isinstance(m,Projected)}
    judgement={name:judge(means[c],means[k]) for name,c,k in comparisons(arm)}
    gain=means['C2MS']['RR']['PSNR']-means['C0']['RR']['PSNR']
    retention=dict(c2ms_gain_dB=gain,arm_share=(means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain,arm_P_share=(means[arm+'+P']['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain)
    wins={f'{c}_vs_{k}':dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[c],rows[k])),
                             SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[c],rows[k]))) for _,c,k in comparisons(arm)}
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    val53=torch.load(paths['C2MS'][0],map_location='cpu',weights_only=False)['metrics'];val65=torch.load(paths['L1S'][0],map_location='cpu',weights_only=False)['metrics']
    validation=dict(arm=best,control_53=val53,control_65=val65,delta_psnr_vs_53=best['val_psnr']-val53['val_psnr'],delta_sam_vs_53=best['val_sam']-val53['val_sam'],
                    delta_psnr_vs_65=best['val_psnr']-val65['val_psnr'],delta_sam_vs_65=best['val_sam']-val65['val_sam'])
    shapes=[(1,1,256,256),(1,bands,256,256),(1,bands,64,64)]
    cost={name:dict(parameters=sum(p.numel() for p in models[name].parameters()),used_parameters=used_parameters(models[name]),conv_gmac_256=conv_macs(models[name],shapes)/1e9) for name in ('C2MS','L1S',arm,'C0')}
    timing=None
    if seed==cfg['latency_seed']:
        with h5py.File(rr) as r,h5py.File(fr) as f:
            xs={'RR_256':[torch.from_numpy((np.asarray(r[k][0],dtype=np.float64)/SENSORS[sensor]['max_value']).astype('float32'))[None].cuda() for k in ('pan','lms','ms')],
                'FR_512':[torch.from_numpy((np.asarray(f[k][0],dtype=np.float64)/SENSORS[sensor]['max_value']).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]}
        timing={size:{name:latency(models[name],x,20 if '+P' not in name else 10) for name in ('C2MS',arm,'C2MS+P',arm+'+P')} for size,x in xs.items()}
        timing['note']='GPU seconds per scene, median; scene 0; the projection runs in float64 conjugate gradient after the network; measured at evaluation time with no training running'
    record=dict(sensor=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)',weights=weights,cost=cost,timing=timing,projection_iterations=iterations,observation_rmse_DN=obs,
                rows=rows,means=means,judgement=judgement,retention=retention,scene_wins=wins,validation=validation,selection=cfg['selection'],time_kst=now())
    write(dest/'evaluation'/f'{sensor}_v2_4.json',record)
    result_images.render(models['C2MS'],models[arm+'+P'],rr.parent,dest/'evaluation'/sensor,names=('C2MS',arm+'+P'),seed=seed,scenes=cfg['scenes'],
        labels=('현재 기준 C2MS',LABELS[arm]),rr_file=rr.name,scale=SENSORS[sensor]['max_value'],bands=SENSORS[sensor]['rgb'])
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam','lr'),('Train mean |error| (original scale, L1)','Validation PSNR (peak 1)','Validation SAM (degrees)','Learning rate')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid]);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    axes[1].axhline(val53['val_psnr'],color='gray',ls='--',lw=1,label='53 C2MS');axes[1].axhline(val65['val_psnr'],color='#d1495b',ls=':',lw=1,label='65 C2MS L1S');axes[1].legend(frameon=False)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; best selected by validation MSE');fig.tight_layout();fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,3.6))
    names=['C0','C2MS+P','L1S','L1S+P',arm,arm+'+P'];labels={'C0':'original C0','C2MS+P':'C2MS + P','L1S':'65 L1S','L1S+P':'65 L1S + P',arm:'arm',arm+'+P':'arm + P'}
    colors={'C0':'#8d99ae','C2MS+P':'#adb5bd','L1S':'#f4a261','L1S+P':'#e9c46a',arm:'#e76f51',arm+'+P':'#d1495b'}
    for ax,key in zip(axes,('PSNR','SAM','ERGAS')):
        ax.bar([labels[n] for n in names],[means[n]['RR'][key]-means['C2MS']['RR'][key] for n in names],color=[colors[n] for n in names]);ax.axhline(0,color='black',lw=.8)
        ax.set(title=f'{sensor} test RR {key} - 53 C2MS');ax.grid(axis='y',alpha=.3);ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(dest/'evaluation/comparison.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,cost=cost,timing=timing,judgement=judgement,retention=retention,validation=validation,
                    scope='full 20-scene RR (judgement) and FR QNR (reference); v2.4; per seed; no automatic baseline promotion'))


def stage_gate(cfg):
    """Stage 1 rule fixed before training: an arm passes when, at seed 42, arm + P beats plain C2MS on all three RR metrics; if both pass
    the one with fewer operations (64 x 12) goes on; if none passes, stage 2 is not run."""
    runtime=Path(cfg['runtime']);rows={}
    for arm in sorted(SHAPES,key=lambda a:SHAPES[a][1]):
        j=json.loads((runtime/'runs'/run_name('QB',arm,cfg['latency_seed'])/'evaluation_complete.json').read_text(encoding='utf8'))['judgement']['official_arm_P_vs_53']
        rows[arm]=dict(delta_RR=j['delta_RR'],passed=all(j['rr_components_improved'].values()))
    chosen=next((a for a in rows if rows[a]['passed']),None)
    return dict(time_kst=now(),rule=stage_gate.__doc__,arms=rows,chosen=chosen)


def summarize(cfg,arm,seeds):
    """Average over seeds of the same-seed paired comparisons (criteria v2.4: 0.04 dB threshold for two or more seeds)."""
    runtime=Path(cfg['runtime']);per={}
    for s in seeds:per[s]=json.loads((runtime/'runs'/run_name('QB',arm,s)/'evaluation'/'QB_v2_4.json').read_text(encoding='utf8'))
    names=list(per[seeds[0]]['means']);mean={n:{part:{k:float(np.mean([per[s]['means'][n][part][k] for s in seeds])) for k in per[seeds[0]]['means'][n][part]} for part in ('RR','FR')} for n in names}
    judgement={name:judge(mean[c],mean[k],len(seeds)) for name,c,k in comparisons(arm)}
    per_seed={s:{name:{k:per[s]['judgement'][name]['delta_RR'][k] for k in ('PSNR','SAM','ERGAS')} for name,_,_ in comparisons(arm)} for s in seeds}
    out=dict(time_kst=now(),arm=arm,seeds=seeds,means=mean,judgement=judgement,per_seed_delta_RR=per_seed,criteria='v2.4, mean of same-seed pairs')
    write(runtime/'summary.json',out);return out
