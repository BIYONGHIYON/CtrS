"""Evaluate each validation-selected run (from 80 evaluate.py) on the 20 RR/FR test scenes of its sensor (WV3 runs also on WV2, reference).

Systems (same seed and sensor): the arm and the arm with the K10 projection (`+P`, prepare.Projected); the official control C2MS (QB 53,
GF2/WV3 54) with and without the projection; 65 C2MS_L1S (the weights the arm was factorized from) with and without the projection; the
original SSA-MRN C0. The controls do not depend on the arm, so they are computed once per (sensor, seed, target) and cached in
runtime/controls/. Only RR (judgement, v2.4) and FR QNR/D_lambda/D_s (reference) are computed; RR2/FR_WALD were dropped in v2.4.
stage_gate() uses validation only (no test); summarize() averages same-seed paired test comparisons over seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,M54,conv_macs,used_parameters,ARMS
from prepare import Wald,SENSORS,data_files,Projected
from train import T,sha,write,now,run_name
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
from common import result_images
from common.metrics import rr_metrics,fr_metrics
from common import evaluation_protocol as protocol
LABELS={'FT0':'65번 + 미세조정 + 사영','K1':'K1 분해 + 미세조정 + 사영','K2':'K2 분해 + 미세조정 + 사영','K3':'K3 분해 + 미세조정 + 사영','K2L':'K2 분해 + 40에폭 미세조정 + 사영'}
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4: single seed 0.06 dB, two or more seeds 0.04 dB
TRANSFER={'WV3':['WV2']}
CONTROLS=('C2MS','C2MS+P','L1S','L1S+P','C0')


def targets(sensor):return [sensor]+TRANSFER.get(sensor,[])


def judge(cand,ctrl,seeds=1):
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def comparisons(arm):
    """(name, candidate, control): the official one first."""
    return [('official_arm_P_vs_C2MS',arm+'+P','C2MS'),('arm_vs_C2MS',arm,'C2MS'),('arm_P_vs_C2MS_P',arm+'+P','C2MS+P'),
            ('vs_65_raw',arm,'L1S'),('vs_65_P',arm+'+P','L1S+P'),('reference_65_P_vs_C2MS','L1S+P','C2MS'),('reference_arm_vs_original',arm,'C0')]


def latency(net,x,repeats):
    with torch.inference_mode():
        for _ in range(3):net(*x)
        torch.cuda.synchronize();times=[]
        for _ in range(repeats):
            t=time.perf_counter();net(*x);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    return float(np.median(times))


def expected_sha(cfg,target):return (cfg['data_sha256'] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'])[target]


def scene(net,a,b,wald):
    s=wald.max_value;rr=rr_metrics(a['gt'],V.predict(net,a['pan'],a['lms'],a['ms'],s,'cuda'))
    fused=V.predict(net,b['pan'],b['lms'],b['ms'],s,'cuda');fr=fr_metrics(fused,b['lms'],b['ms'],b['pan'])
    return dict(RR=rr,FR=fr,OBS_RMSE_DN=wald.observation_rmse(fused,b['ms'],'cuda',protocol.OBS_MARGIN))


def run_systems(cfg,target,models):
    rr=Path(cfg['data_root'])/data_files(target)['RR'];fr=Path(cfg['data_root'])/data_files(target)['FR'];want=expected_sha(cfg,target)
    if sha(rr)!=want['RR'] or sha(fr)!=want['FR']:raise ValueError('test data mismatch '+target)
    wald=Wald(target,'cuda');rows={n:[] for n in models}
    with torch.inference_mode(),h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for n,m in models.items():rows[n].append(dict(index=i,metrics=scene(m,a,b,wald)))
            print(json.dumps(dict(time_kst=now(),target=target,scene=i,systems=list(models))),flush=True)
    iters={n:dict(mean=float(np.mean(m.iterations)),max=int(max(m.iterations))) for n,m in models.items() if isinstance(m,Projected)}
    return rows,iters


def load(path,kind,bands,seed,arm=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=M54.Model(kind,bands,seed) if kind in ('C0','C2MS') else Model(arm,bands,seed)
    m.load_state_dict(ck['model'],strict=True);return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'))


def control_rows(cfg,sensor,target,seed):
    """Cached control systems for (sensor, seed, target)."""
    root=HERE.parents[2];bands=SENSORS[sensor]['bands'];p=Path(cfg['runtime'])/'controls'/f'{sensor}_s{seed}_{target}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    seeded=lambda key:root/cfg[key][sensor].replace('<seed>',str(seed))
    c2,w1=load(seeded('controls'),'C2MS',bands,seed);l1,w2=load(seeded('init_from'),'C2MS',bands,seed);c0,w3=load(seeded('original'),'C0',bands,seed)
    models={'C2MS':c2,'C2MS+P':Projected(c2,target),'L1S':l1,'L1S+P':Projected(l1,target),'C0':c0}
    rows,iters=run_systems(cfg,target,models);rec=dict(rows=rows,projection_iterations=iters,weights=dict(C2MS=w1,L1S=w2,C0=w3),time_kst=now())
    write(p,rec);return rec


def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    bands=SENSORS[sensor]['bands'];net,wa=load(dest/'checkpoints/best.pt',arm,bands,seed,arm);records={}
    for target in targets(sensor):
        ctrl=control_rows(cfg,sensor,target,seed);models={arm:net,arm+'+P':Projected(net,target)}
        rows,iters=run_systems(cfg,target,models);rows.update(ctrl['rows']);iters.update(ctrl['projection_iterations'])
        means={n:{part:{k:float(np.mean([x['metrics'][part][k] for x in v])) for k in v[0]['metrics'][part]} for part in ('RR','FR')} for n,v in rows.items()}
        obs={n:float(np.mean([x['metrics']['OBS_RMSE_DN'] for x in v])) for n,v in rows.items()}
        judgement={name:judge(means[c],means[k]) for name,c,k in comparisons(arm)}
        gain=means['C2MS']['RR']['PSNR']-means['C0']['RR']['PSNR']
        retention=dict(c2ms_gain_dB=gain,arm_share=(means[arm]['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain,arm_P_share=(means[arm+'+P']['RR']['PSNR']-means['C0']['RR']['PSNR'])/gain)
        wins={name:dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[c],rows[k])),SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[c],rows[k]))) for name,c,k in comparisons(arm)}
        timing=None;rr=Path(cfg['data_root'])/data_files(target)['RR']
        if seed==cfg['latency_seed'] and target==sensor:
            mv=SENSORS[target]['max_value'];fr=Path(cfg['data_root'])/data_files(target)['FR']
            c2,_=load(HERE.parents[2]/cfg['controls'][sensor].replace('<seed>',str(seed)),'C2MS',bands,seed)
            timed={'C2MS':c2,arm:net,'C2MS+P':Projected(c2,target),arm+'+P':Projected(net,target)}
            with h5py.File(rr) as r,h5py.File(fr) as f:
                xs={'RR_256':[torch.from_numpy((np.asarray(r[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')],
                    'FR_512':[torch.from_numpy((np.asarray(f[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]}
            timing={size:{n:latency(m,x,20 if '+P' not in n else 10) for n,m in timed.items()} for size,x in xs.items()}
            timing['note']='GPU seconds per scene, median, scene 0; K10 projection; measured with no training on the GPU'
            result_images.render(timed['C2MS'],timed[arm+'+P'],rr.parent,dest/'evaluation'/target,names=('C2MS',arm+'+P'),seed=seed,scenes=cfg['scenes'],
                labels=('현재 기준 C2MS',LABELS[arm]),rr_file=rr.name,scale=SENSORS[target]['max_value'],bands=SENSORS[target]['rgb'],land_cover={i:f'{target} 장면 {i}' for i in cfg['scenes']})
        records[target]=dict(target=target,training_sensor=sensor,transfer=target!=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)'+('; transfer test, reference only' if target!=sensor else ''),
                             timing=timing,projection_iterations=iters,observation_rmse_DN=obs,rows=rows,means=means,judgement=judgement,retention=retention,scene_wins=wins,selection=cfg['selection'],time_kst=now())
        write(dest/'evaluation'/f'{target}_v2_4.json',records[target])
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    vc=torch.load(HERE.parents[2]/cfg['controls'][sensor].replace('<seed>',str(seed)),map_location='cpu',weights_only=False)['metrics']
    v65=torch.load(HERE.parents[2]/cfg['init_from'][sensor].replace('<seed>',str(seed)),map_location='cpu',weights_only=False)['metrics']
    validation=dict(arm=best,control_C2MS=vc,control_65=v65,delta_psnr_vs_C2MS=best['val_psnr']-vc['val_psnr'],delta_sam_vs_C2MS=best['val_sam']-vc['val_sam'],
                    delta_psnr_vs_65=best['val_psnr']-v65['val_psnr'],delta_sam_vs_65=best['val_sam']-v65['val_sam'],epoch0=history[0])
    shapes=[(1,1,256,256),(1,bands,256,256),(1,bands,64,64)]
    with torch.inference_mode():cost=dict(parameters=sum(p.numel() for p in net.parameters()),used_parameters=used_parameters(net),conv_gmac_256=conv_macs(net,shapes)/1e9)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam'),('Train mean |error| (L1)','Validation PSNR (peak 1)','Validation SAM (degrees)')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid],marker='o');ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    for ax,key in zip(axes[1:],('val_psnr','val_sam')):ax.axhline(vc[key],color='gray',ls='--',lw=1,label='C2MS');ax.axhline(v65[key],color='#d1495b',ls=':',lw=1,label='65 C2MS L1S');ax.legend(frameon=False)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; epoch 0 = right after factorization; best by validation MSE');fig.tight_layout();(dest/'evaluation').mkdir(exist_ok=True);fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=wa,cost=cost,validation=validation,targets={t:dict(judgement=r['judgement'],retention=r['retention'],timing=r['timing'],transfer=r['transfer']) for t,r in records.items()},
                    scope='full 20-scene RR (judgement on the training sensor) and FR QNR (reference); WV2 transfer reference only; v2.4; per seed; K10 projection; no automatic baseline promotion'))


def stage_gate(cfg):
    """Validation only (fixed before training): on every stage-1 sensor, the selected weights must reach validation PSNR >= 65 - 0.04 dB and
    SAM <= 65 + 0.02 deg (65 = the weights the arm was factorized from, same seed). The lightest passing arm is chosen in the order
    K3, K2, K2L, K1; FT0 is a control and never chosen."""
    runtime=Path(cfg['runtime']);seed=cfg['latency_seed'];rows={};passed={}
    for arm in ARMS:
        rows[arm]={}
        for sensor in cfg['stage1_sensors']:
            v=json.loads((runtime/'runs'/run_name(sensor,arm,seed)/'evaluation_complete.json').read_text(encoding='utf8'))['validation']
            rows[arm][sensor]=dict(dpsnr_vs_65=v['delta_psnr_vs_65'],dsam_vs_65=v['delta_sam_vs_65'],best_epoch=v['arm'].get('epoch'))
        passed[arm]=all(r['dpsnr_vs_65']>=cfg['gate']['val_psnr_min'] and r['dsam_vs_65']<=cfg['gate']['val_sam_max'] for r in rows[arm].values())
    chosen=next((a for a in cfg['gate']['order'] if passed[a]),None)
    return dict(time_kst=now(),rule=cfg['gate'],validation=rows,passed=passed,chosen=chosen)


def summarize(cfg,arm,seeds):
    runtime=Path(cfg['runtime']);out=dict(time_kst=now(),arm=arm,seeds=seeds,criteria='v2.4, mean of same-seed pairs; WV2 transfer reference only',targets={})
    for sensor in cfg['sensors']:
        for target in targets(sensor):
            per={s:json.loads((runtime/'runs'/run_name(sensor,arm,s)/'evaluation'/f'{target}_v2_4.json').read_text(encoding='utf8')) for s in seeds}
            names=list(per[seeds[0]]['means'])
            mean={n:{part:{k:float(np.mean([per[s]['means'][n][part][k] for s in seeds])) for k in per[seeds[0]]['means'][n][part]} for part in ('RR','FR')} for n in names}
            judgement={name:judge(mean[c],mean[k],len(seeds)) for name,c,k in comparisons(arm)}
            per_seed={s:{name:{k:per[s]['judgement'][name]['delta_RR'][k] for k in ('PSNR','SAM','ERGAS')} for name,_,_ in comparisons(arm)} for s in seeds}
            wins={name:{k:sum(per[s]['scene_wins'][name][k] for s in seeds) for k in ('PSNR','SAM')} for name in per[seeds[0]]['scene_wins']}
            out['targets'][target]=dict(training_sensor=sensor,transfer=target!=sensor,means=mean,judgement=judgement,per_seed_delta_RR=per_seed,scene_wins=wins)
    write(runtime/'summary.json',out);return out
