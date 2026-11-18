"""88 (from 87 evaluate.py). Evaluate each validation-selected run on the 20 RR/FR test scenes of its sensor; WV3 runs also on WV2 (transfer,
reference only, with the 82 per-band calibration as for every C2MS-SP model on an untrained sensor).

Systems (same seed and sensor): the arm and the arm + K10 projection (`+P`, the 86 FastProjected); the official control C2MS-SP = 65 C2MS_L1S
+ the same projection (`L1S+P`) and 65 without projection (`L1S`); references C2MS-SP-Lite = 83 K2 (`Lite`, `Lite+P`) and the 84 B150
weights (`B150`, `B150+P`). Controls are computed once per (sensor, seed, target) and cached.
Only RR (judgement, v2.4) and FR QNR/D_lambda/D_s (reference) are computed. summarize_arms() reports the single-seed paired test comparisons
(official: K2D + P vs C2MS-SP); measure() is the fixed latency protocol (as 85)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import Model,M54,M83,conv_macs,used_parameters,ARMS,FACTORIZED
from prepare import Wald,SENSORS,data_files,Projected,FastProjected
from train import T,sha,write,now,run_name
def _mod(name,rel):
    spec=importlib.util.spec_from_file_location(name,HERE.parent/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
V=_mod('c2ms54_eval','54_multi_sensor_scale_mixed_training/evaluate.py')
C82=_mod('exp82_prepare','82_cross_sensor_band_calibration/prepare.py')
from common import result_images
from common.metrics import rr_metrics,fr_metrics
from common import evaluation_protocol as protocol
LABELS={'TRUNK':'C2MS 학습률 고정 120에폭 + 사영','FULL':'C2MS 120 + 감쇠 30에폭 + 사영','K2D':'감쇠 직전 K2 분해(150에폭) + 사영','K2F':'감쇠 후 K2 분해 + 10에폭 + 사영'}
CONTROLS=(('L1S','init_from'),('Lite','lite'),('B150','reference_84'))
OFFICIAL='K2D'
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4: single seed 0.06 dB, two or more seeds 0.04 dB
TRANSFER={'WV3':['WV2']}


class Calibrated(C82.Calibrated):
    """82 calibration ('scene') with its reference ratios moved to the input device (82 ran on the CPU; as 84 evaluate.py)."""
    def forward(self,pan,lms,ms):
        self.r_train=self.r_train.to(pan.device);return super().forward(pan,lms,ms)


def targets(sensor):return [sensor]+TRANSFER.get(sensor,[])


def projected(cfg,net,target):return FastProjected(net,target)


def systems(cfg,net,sensor,target,name):
    """{name: net, name+'+P': projected} on the training sensor; on a transfer target both are wrapped in the 82 calibration first."""
    if target!=sensor:
        ref=json.loads((HERE.parent/'82_cross_sensor_band_calibration'/'results/reference_stats'/f'{sensor}.json').read_text(encoding='utf8'));net=Calibrated(net,ref,'scene')
    return {name:net,name+'+P':projected(cfg,net,target)}


def judge(cand,ctrl,seeds=1):
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def comparisons(arm,b150=True):
    """(name, candidate, control): vs C2MS-SP first (official for K2D, reference for the other arms); the 84 B150 control exists for seed 42 only."""
    return [('arm_P_vs_C2MS_SP',arm+'+P','L1S+P'),('arm_P_vs_Lite_P',arm+'+P','Lite+P')]+([('arm_P_vs_84_B150_P',arm+'+P','B150+P')] if b150 else [])+[('arm_vs_65_raw',arm,'L1S')]


def has_b150(cfg,seed):return seed in cfg.get('reference_84_seeds',[42])


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
    """arm in ARMS, or 'L1S'/'B150' (C2MS weights of 65/84) or 'Lite' (83 K2)."""
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=Model('K2' if arm=='Lite' else arm,bands,seed)
    m.load_state_dict(ck['model'],strict=True);return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'))


def seeded(cfg,key,sensor,seed):return HERE.parents[2]/cfg[key][sensor].replace('<seed>',str(seed))


def control_rows(cfg,sensor,target,seed):
    """Cached control systems for (sensor, seed, target), each without and with the projection: C2MS-SP (65), C2MS-SP-Lite (83 K2), 84 B150."""
    p=Path(cfg['runtime'])/'controls'/f'{sensor}_s{seed}_{target}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    models={};w={}
    for name,key in CONTROLS:
        if name=='B150' and not has_b150(cfg,seed):continue
        net,w[name]=load(seeded(cfg,key,sensor,seed),name,SENSORS[sensor]['bands'],seed);models.update(systems(cfg,net,sensor,target,name))
    rows,iters=run_systems(cfg,target,models);rec=dict(rows=rows,projection_iterations=iters,weights=w,projection='fast (86)',time_kst=now())
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
        cmp=comparisons(arm,has_b150(cfg,seed));judgement={name:judge(means[c],means[k]) for name,c,k in cmp}
        wins={name:dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[c],rows[k])),SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[c],rows[k]))) for name,c,k in cmp}
        rr=Path(cfg['data_root'])/data_files(target)['RR']
        if seed==cfg['latency_seed'] and target==sensor:
            l1,_=load(seeded(cfg,'init_from',sensor,seed),'L1S',bands,seed)
            result_images.render(projected(cfg,l1,target),models[arm+'+P'],rr.parent,dest/'evaluation'/target,names=('C2MS-SP',arm+'+P'),seed=seed,scenes=cfg['scenes'],
                labels=('현재 기준 C2MS-SP',LABELS[arm]),rr_file=rr.name,scale=SENSORS[target]['max_value'],bands=SENSORS[target]['rgb'],land_cover={i:f'{target} 장면 {i}' for i in cfg['scenes']})
        records[target]=dict(target=target,training_sensor=sensor,transfer=target!=sensor,arm=arm,seed=seed,criteria='v2.4 (RR only)'+('; transfer test with the 82 calibration, reference only' if target!=sensor else ''),
                             projection='fast (86)',projection_iterations=iters,observation_rmse_DN=obs,rows=rows,means=means,judgement=judgement,scene_wins=wins,selection=cfg['selection'],time_kst=now())
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
    if arm in FACTORIZED:axes[1].set_title('epoch 0 = right after the K2 factorization',fontsize=9)
    fig.suptitle(f'{sensor} {arm}, seed {seed}; best by validation MSE');fig.tight_layout();(dest/'evaluation').mkdir(exist_ok=True);fig.savefig(dest/'evaluation/learning.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=wa,cost=cost,validation=validation,targets={t:dict(judgement=r['judgement'],transfer=r['transfer']) for t,r in records.items()},
                    scope='full 20-scene RR (judgement on the training sensor) and FR QNR (reference); WV2 transfer reference only; v2.4; per seed; no automatic baseline promotion'))


def summarize(cfg,arm,seeds,roots=None,name='summary.json'):
    """Mean of the same-seed pairs over `seeds`; roots maps a seed to the runtime holding its runs (default cfg['runtime'])."""
    runtime=Path(cfg['runtime']);roots=roots or {};cmp=comparisons(arm,all(has_b150(cfg,s) for s in seeds))
    out=dict(time_kst=now(),arm=arm,seeds=seeds,criteria='v2.4, mean of same-seed pairs; WV2 transfer reference only',runtimes={s:str(roots.get(s,runtime)) for s in seeds},targets={})
    for sensor in cfg['sensors']:
        for target in targets(sensor):
            per={s:json.loads((Path(roots.get(s,runtime))/'runs'/run_name(sensor,arm,s)/'evaluation'/f'{target}_v2_4.json').read_text(encoding='utf8')) for s in seeds}
            names=[n for n in per[seeds[0]]['means'] if all(n in per[s]['means'] for s in seeds)]
            mean={n:{part:{k:float(np.mean([per[s]['means'][n][part][k] for s in seeds])) for k in per[seeds[0]]['means'][n][part]} for part in ('RR','FR')} for n in names}
            judgement={nm:judge(mean[c],mean[k],len(seeds)) for nm,c,k in cmp}
            per_seed={s:{nm:{k:per[s]['judgement'][nm]['delta_RR'][k] for k in ('PSNR','SAM','ERGAS')} for nm,_,_ in cmp} for s in seeds}
            wins={nm:{k:sum(per[s]['scene_wins'][nm][k] for s in seeds) for k in ('PSNR','SAM')} for nm,_,_ in cmp}
            out['targets'][target]=dict(training_sensor=sensor,transfer=target!=sensor,means=mean,judgement=judgement,per_seed_delta_RR=per_seed,scene_wins=wins)
    write(runtime/name,out);return out


def summarize_seeds(cfg):
    """Stage 2: K2D over seeds 42 (stage-1 runtime), 43 and 44 (this runtime). Official: K2D + P vs C2MS-SP (v2.4, 0.04 dB), an improvement
    only if QB, GF2 and WV3 all are; K2D + P vs Lite + P reference."""
    s2=cfg['stage2'];seeds=[cfg['latency_seed']]+list(s2['seeds']);roots={cfg['latency_seed']:Path(s2['seed42_runtime'])}
    s=summarize(cfg,OFFICIAL,seeds,roots,'summary_3seeds.json')
    s['official_label']={t:v['judgement']['arm_P_vs_C2MS_SP']['label'] for t,v in s['targets'].items()}
    s['all_training_sensors_improved']=all(v['judgement']['arm_P_vs_C2MS_SP']['label']=='개선' for v in s['targets'].values() if not v['transfer'])
    s['light_non_inferior']=all(v['judgement']['arm_P_vs_C2MS_SP']['delta_RR']['PSNR']>=-0.04 for v in s['targets'].values() if not v['transfer'])
    write(Path(cfg['runtime'])/'summary_3seeds.json',s);return s


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


def summarize_arms(cfg):
    """Single seed: every arm on every training sensor vs C2MS-SP, Lite and 84 B150, and the WV2 transfer (reference). Official: K2D + P vs
    C2MS-SP (v2.4, 0.06 dB), an improvement only if QB, GF2 and WV3 all are; light non-inferior = K2D + P within -0.04 dB on every sensor
    (reported, not a judgement). Pairs between runs: K2D vs FULL (factorization cost) and K2F vs K2D (factorize before or after the decay)."""
    seed=cfg['latency_seed'];runtime=Path(cfg['runtime'])
    out=dict(time_kst=now(),seed=seed,official=OFFICIAL,criteria='v2.4, single seed (0.06 dB); K2D + P vs C2MS-SP is official and an improvement only if QB, GF2 and WV3 all are; WV2 reference only',arms={},pairs={})
    for arm in ARMS:
        s=summarize(cfg,arm,[seed])
        out['arms'][arm]=dict(targets={t:dict(judgement=v['judgement'],means={k:v['means'][k] for k in (arm,arm+'+P','L1S','L1S+P','Lite','Lite+P','B150','B150+P')},scene_wins=v['scene_wins'],transfer=v['transfer']) for t,v in s['targets'].items()},
                              all_training_sensors_improved=all(v['judgement']['arm_P_vs_C2MS_SP']['label']=='개선' for v in s['targets'].values() if not v['transfer']),
                              light_non_inferior=all(v['judgement']['arm_P_vs_C2MS_SP']['delta_RR']['PSNR']>=-0.04 for v in s['targets'].values() if not v['transfer']))
    for name,a,b in (('K2D_vs_FULL','K2D','FULL'),('K2F_vs_K2D','K2F','K2D'),('FULL_vs_84_B150','FULL','B150')):
        out['pairs'][name]={}
        for sensor in cfg['sensors']:
            for target in targets(sensor):
                ra=json.loads((runtime/'runs'/run_name(sensor,a,seed)/'evaluation'/f'{target}_v2_4.json').read_text(encoding='utf8'))
                rb=ra if b=='B150' else json.loads((runtime/'runs'/run_name(sensor,b,seed)/'evaluation'/f'{target}_v2_4.json').read_text(encoding='utf8'))
                out['pairs'][name][target]=judge(ra['means'][a+'+P'],rb['means'][b+'+P'])
    out['official_label']={t:v['judgement']['arm_P_vs_C2MS_SP']['label'] for t,v in out['arms'][OFFICIAL]['targets'].items()}
    write(runtime/'summary.json',out);return out
