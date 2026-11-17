"""Evaluate each validation-selected run (from 83 evaluate.py) on the 20 RR/FR test scenes of its sensor; WV3 runs also on WV2 (transfer,
reference only, with the 82 per-band calibration as for every C2MS-SP model on an untrained sensor).

Systems (same seed and sensor): the arm and the arm + K10 projection (`+P`); the official control C2MS-SP = 65 C2MS_L1S + the same projection
(`L1S+P`) and 65 without projection (`L1S`). The projection is prepare.FastProjected when run.py stage0 found it equal to the 79/83 projection
on every test scene (cfg['projection'] == 'fast'), else the 83 Projected. Controls are computed once per (sensor, seed, target) and cached.
Only RR (judgement, v2.4) and FR QNR/D_lambda/D_s (reference) are computed. stage_gate() uses validation only (no test); summarize()
averages same-seed paired test comparisons; measure() is the fixed latency protocol of the report."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,M54,M83,conv_macs,used_parameters,ARMS
from prepare import Wald,SENSORS,data_files,Projected,FastProjected
from train import T,sha,write,now,run_name
def _mod(name,rel):
    spec=importlib.util.spec_from_file_location(name,HERE.parent/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
V=_mod('c2ms54_eval','54_multi_sensor_scale_mixed_training/evaluate.py')
C82=_mod('exp82_prepare','82_cross_sensor_band_calibration/prepare.py')
from common import result_images
from common.metrics import rr_metrics,fr_metrics
from common import evaluation_protocol as protocol
LABELS={'FT0L':'65번 + 40에폭 미세조정 + 사영','B64W96':'B64W96 + 사영','B32W96':'B32W96 + 사영','B32W64':'B32W64 + 사영'}
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4: single seed 0.06 dB, two or more seeds 0.04 dB
TRANSFER={'WV3':['WV2']}


def targets(sensor):return [sensor]+TRANSFER.get(sensor,[])


def projected(cfg,net,target):return FastProjected(net,target) if cfg.get('projection')=='fast' else Projected(net,target)


def systems(cfg,net,sensor,target,name):
    """{name: net, name+'+P': projected} on the training sensor; on a transfer target both are wrapped in the 82 calibration first."""
    if target!=sensor:
        ref=json.loads((HERE.parent/'82_cross_sensor_band_calibration'/'results/reference_stats'/f'{sensor}.json').read_text(encoding='utf8'));net=C82.Calibrated(net,ref,'scene')
    return {name:net,name+'+P':projected(cfg,net,target)}


def judge(cand,ctrl,seeds=1):
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def comparisons(arm):
    """(name, candidate, control): the official one first."""
    return [('official_arm_P_vs_C2MS_SP',arm+'+P','L1S+P'),('arm_vs_65_raw',arm,'L1S')]


def scene(net,a,b,wald):
    s=wald.max_value;rr=rr_metrics(a['gt'],V.predict(net,a['pan'],a['lms'],a['ms'],s,'cuda'))
    fused=V.predict(net,b['pan'],b['lms'],b['ms'],s,'cuda');fr=fr_metrics(fused,b['lms'],b['ms'],b['pan'])
    return dict(RR=rr,FR=fr,OBS_RMSE_DN=wald.observation_rmse(fused,b['ms'],'cuda',protocol.OBS_MARGIN))


def expected_sha(cfg,target):return (cfg['data_sha256'] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'])[target]


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
    iters={n:dict(mean=float(np.mean(m.iterations)),max=int(max(m.iterations))) for n,m in models.items() if isinstance(m,(Projected,FastProjected))}
    return rows,iters


def load(path,arm,bands,seed):
    """arm in ARMS (85 structure) or 'L1S' (65 C2MS_L1S) or 'K2' (83 C2MS-SP-Lite)."""
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=M54.Model('C2MS',bands,seed) if arm=='L1S' else M83.Model('K2',bands,seed) if arm=='K2' else Model(arm,bands,seed)
    m.load_state_dict(ck['model'],strict=True);return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'))


def seeded(cfg,key,sensor,seed):return HERE.parents[2]/cfg[key][sensor].replace('<seed>',str(seed))


def control_rows(cfg,sensor,target,seed):
    """Cached control systems for (sensor, seed, target): 65 C2MS_L1S without and with the projection (C2MS-SP)."""
    p=Path(cfg['runtime'])/'controls'/f'{sensor}_s{seed}_{target}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    l1,w=load(seeded(cfg,'init_from',sensor,seed),'L1S',SENSORS[sensor]['bands'],seed)
    rows,iters=run_systems(cfg,target,systems(cfg,l1,sensor,target,'L1S'));rec=dict(rows=rows,projection_iterations=iters,weights=dict(L1S=w),projection=cfg.get('projection'),time_kst=now())
    write(p,rec);return rec


def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);runtime=Path(cfg['runtime']);dest=runtime/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    bands=SENSORS[sensor]['bands'];net,wa=load(dest/'checkpoints/best.pt',arm,bands,seed);records={}
    for target in targets(sensor):
        ctrl=control_rows(cfg,sensor,target,seed);models=systems(cfg,net,sensor,target,arm)
        rows,iters=run_systems(cfg,target,models);rows.update(ctrl['rows']);iters.update(ctrl['projection_iterations'])
        means={n:{part:{k:float(np.mean([x['metrics'][part][k] for x in v])) for k in v[0]['metrics'][part]} for part in ('RR','FR')} for n,v in rows.items()}
        obs={n:float(np.mean([x['metrics']['OBS_RMSE_DN'] for x in v])) for n,v in rows.items()}
        judgement={name:judge(means[c],means[k]) for name,c,k in comparisons(arm)}
        wins={name:dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[c],rows[k])),SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[c],rows[k]))) for name,c,k in comparisons(arm)}
        rr=Path(cfg['data_root'])/data_files(target)['RR']
        if seed==cfg['latency_seed'] and target==sensor:
            l1,_=load(seeded(cfg,'init_from',sensor,seed),'L1S',bands,seed)
            result_images.render(projected(cfg,l1,target),models[arm+'+P'],rr.parent,dest/'evaluation'/target,names=('C2MS-SP',arm+'+P'),seed=seed,scenes=cfg['scenes'],
                labels=('현재 기준 C2MS-SP',LABELS[arm]),rr_file=rr.name,scale=SENSORS[target]['max_value'],bands=SENSORS[target]['rgb'],land_cover={i:f'{target} 장면 {i}' for i in cfg['scenes']})
        records[target]=dict(target=target,training_sensor=sensor,transfer=target!=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)'+('; transfer test with the 82 calibration, reference only' if target!=sensor else ''),
                             projection=cfg.get('projection'),projection_iterations=iters,observation_rmse_DN=obs,rows=rows,means=means,judgement=judgement,scene_wins=wins,selection=cfg['selection'],time_kst=now())
        write(dest/'evaluation'/f'{target}_v2_4.json',records[target])
    history=json.loads((dest/'history.json').read_text(encoding='utf8'));best=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))['best_val']
    v65=torch.load(seeded(cfg,'init_from',sensor,seed),map_location='cpu',weights_only=False)['metrics']
    validation=dict(arm=best,control_65=v65,delta_psnr_vs_65=best['val_psnr']-v65['val_psnr'],delta_sam_vs_65=best['val_sam']-v65['val_sam'],epoch0=history[0])
    shapes=[(1,1,256,256),(1,bands,256,256),(1,bands,64,64)]
    with torch.inference_mode():cost=dict(parameters=sum(p.numel() for p in net.parameters()),used_parameters=used_parameters(net),conv_gmac_256=conv_macs(net,shapes)/1e9)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    for ax,key,label in zip(axes,('train_mse','val_psnr','val_sam'),('Train mean |error| (L1)','Validation PSNR (peak 1)','Validation SAM (degrees)')):
        valid=[x for x in history if x.get(key) is not None];ax.plot([x['epoch'] for x in valid],[x[key] for x in valid],marker='o',ms=3);ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.3)
    for ax,key in zip(axes[1:],('val_psnr','val_sam')):ax.axhline(v65[key],color='#d1495b',ls=':',lw=1,label='65 C2MS L1S');ax.legend(frameon=False)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; epoch 0 = right after the channel reduction; best by validation MSE');fig.tight_layout();(dest/'evaluation').mkdir(exist_ok=True);fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=wa,cost=cost,validation=validation,targets={t:dict(judgement=r['judgement'],transfer=r['transfer']) for t,r in records.items()},
                    scope='full 20-scene RR (judgement on the training sensor) and FR QNR (reference); WV2 transfer reference only; v2.4; per seed; no automatic baseline promotion'))


def stage_gate(cfg):
    """Validation only (fixed before training): on every stage-1 sensor the selected weights must reach validation PSNR >= 65 and SAM <= 65
    (65 = the same-seed weights the arm started from). Among the passing trained arms the one with the shortest stage-0 network time is chosen
    (cfg['gate']['order']); FT0L is a control and never chosen."""
    runtime=Path(cfg['runtime']);seed=cfg['latency_seed'];rows={};passed={}
    for arm in cfg['trained_arms']:
        rows[arm]={}
        for sensor in cfg['stage1_sensors']:
            v=json.loads((runtime/'runs'/run_name(sensor,arm,seed)/'evaluation_complete.json').read_text(encoding='utf8'))['validation']
            rows[arm][sensor]=dict(dpsnr_vs_65=v['delta_psnr_vs_65'],dsam_vs_65=v['delta_sam_vs_65'],best_epoch=v['arm'].get('epoch'))
        passed[arm]=all(r['dpsnr_vs_65']>=cfg['gate']['val_psnr_min'] and r['dsam_vs_65']<=cfg['gate']['val_sam_max'] for r in rows[arm].values())
    chosen=next((a for a in cfg['gate']['order'] if passed.get(a)),None)
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


def _inputs(cfg,sensor):
    mv=SENSORS[sensor]['max_value'];out={}
    for split,size in (('RR','256'),('FR','512')):
        with h5py.File(Path(cfg['data_root'])/data_files(sensor)[split]) as f:
            out[f'{split}_{size}']=[torch.from_numpy((np.asarray(f[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]
    return out


def measure(cfg,nets,label,rounds=3,warmup=5,repeats=50,setting='official',fp16=False):
    """Fixed latency protocol: one process, batch 1, test scene 0 (RR 256^2, FR 512^2) per sensor, systems interleaved; per round the median
    of `repeats` runs after `warmup`; the reported value is the median of the round medians (seconds). nets: {sensor: {name: module}}.
    setting 'official' = float32, TF32 off, cuDNN deterministic, no autotune (as 81); 'optimized' = cuDNN autotune and TF32 on (reference)."""
    det,bench,tf32=(True,False,False) if setting=='official' else (False,True,True)
    torch.backends.cudnn.deterministic=det;torch.backends.cudnn.benchmark=bench;torch.backends.cudnn.allow_tf32=tf32;torch.backends.cuda.matmul.allow_tf32=tf32
    torch.use_deterministic_algorithms(det,warn_only=True);result={}
    for sensor,systems_ in nets.items():
        for size,x in _inputs(cfg,sensor).items():
            per={n:[] for n in systems_}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=fp16):
                for n,m in systems_.items():
                    for _ in range(warmup):m(*x)
                torch.cuda.synchronize()
                for _ in range(rounds):
                    times={n:[] for n in systems_}
                    for _ in range(repeats):
                        for n,m in systems_.items():
                            t=time.perf_counter();m(*x);torch.cuda.synchronize();times[n].append(time.perf_counter()-t)
                    for n in systems_:per[n].append(float(np.median(times[n])))
            result[f'{sensor}_{size}']={n:dict(seconds=float(np.median(v)),rounds=v) for n,v in per.items()}
            print(json.dumps(dict(time_kst=now(),measure=label,sensor=sensor,size=size,ms={n:round(1e3*float(np.median(v)),2) for n,v in per.items()})),flush=True)
    for m in (m for s in nets.values() for m in s.values()):
        if hasattr(m,'iterations'):m.iterations.clear()
    return dict(time_kst=now(),label=label,setting=setting,fp16=fp16,rounds=rounds,warmup=warmup,repeats=repeats,gpu=torch.cuda.get_device_name(0),result=result)
