"""Evaluate archived weights with and without the inference projection (no training).

For each training sensor (QB, GF2, WV3) and seed (42-44), on the 20 RR/FR test scenes of the sensor (WV3 models also on WV2, reference):
- C2MS (53 on QB, 54 on GF2/WV3; the official control), C2MS_L1S (65), and the original SSA-MRN C0 (43 on QB, 54 on GF2/WV3);
- C2MS + P and C2MS_L1S + P with the 79 projection (prepare.Projected, float64, tolerance 1e-8, at most 400 iterations);
- C2MS_L1S with the faster projection variants fixed in config.json (float32; float64 capped at 20, 10, 5 iterations).
The official judgement (criteria v2.4, RR only, 3-seed mean, threshold 0.04 dB) is C2MS_L1S + P against plain C2MS of the same seed on QB,
GF2 and WV3; WV2 is reference only. Metrics are the RR and FR parts of the 54/79/80 scene metrics (same prediction, normalization and Wald
operator; RR2 and FR_WALD are not computed). FR QNR/D_lambda/D_s are reference. Seed 42 also records GPU time per variant."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,importlib.util,json,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from prepare import SENSORS,Wald,data_files,Projected,ProjectedVariant
from common import result_images
from common.evaluation_protocol import OBS_MARGIN
def _load(name,rel):
    spec=importlib.util.spec_from_file_location(name,HERE.parent/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
M54=_load('exp54_model','54_multi_sensor_scale_mixed_training/model.py');M43=M54.M43
_path=list(sys.path);V=_load('exp54_eval','54_multi_sensor_scale_mixed_training/evaluate.py');sys.path[:]=_path  # 54 evaluate puts its folder first on sys.path
KST=timezone(timedelta(hours=9));NOISE={1:0.06}
TRANSFER={'WV3':['WV2']}
def now():return datetime.now(KST).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(t,p)
def run_name(sensor,seed):return f'{sensor}_s{seed}'
def targets(sensor):return [sensor]+TRANSFER.get(sensor,[])


def judge(cand,ctrl,seeds=1):
    """Criteria v2.4: RR PSNR, SAM and ERGAS all better and the PSNR gain at or above the noise threshold."""
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,
                noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def comparisons(cfg):
    """(name, candidate, control): the official one first, then the decomposition, then each fast variant."""
    out=[('official_L1S_P_vs_C2MS','L1S+P','C2MS'),('L1S_P_vs_C2MS_P','L1S+P','C2MS+P'),('reference_C2MS_P_vs_C2MS','C2MS+P','C2MS'),
         ('reference_L1S_vs_C2MS','L1S','C2MS'),('reference_L1S_vs_original','L1S','C0')]
    out+=[(f'variant_{v}_vs_C2MS','L1S+'+v,'C2MS') for v in cfg['variants']]
    out+=[(f'variant_{v}_vs_L1S_P','L1S+'+v,'L1S+P') for v in cfg['variants']]
    return out


def scene_metrics(net,rr,fr,wald,device):
    """The RR and FR parts of 54 scene_metrics (same predict, metrics and observation RMSE)."""
    s=wald.max_value
    out={'RR':V.rr_metrics(rr['gt'],V.predict(net,rr['pan'],rr['lms'],rr['ms'],s,device))}
    fused=V.predict(net,fr['pan'],fr['lms'],fr['ms'],s,device)
    out['FR']=V.fr_metrics(fused,fr['lms'],fr['ms'],fr['pan']);out['OBS_RMSE_DN']=wald.observation_rmse(fused,fr['ms'],device,OBS_MARGIN)
    return out


def latency(net,x,repeats):
    """Median seconds per forward (CUDA synchronized, 3 warm-up passes); inputs already on the GPU."""
    with torch.no_grad():
        for _ in range(3):net(*x)
        torch.cuda.synchronize();times=[]
        for _ in range(repeats):
            t=time.perf_counter();net(*x);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    return float(np.median(times))


def load_models(cfg,sensor,seed,device):
    bands=SENSORS[sensor]['bands'];seeded=lambda p:ROOT/p.replace('<seed>',str(seed));nets={};weights={}
    for name,key in (('C2MS','controls'),('L1S','candidates'),('C0','original')):
        path=seeded(cfg[key][sensor]);ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['seed']!=seed:raise ValueError('seed mismatch '+name)
        m=(M43.Model('C0',seed) if sensor=='QB' else M54.Model('C0',bands,seed)) if name=='C0' else M54.Model('C2MS',bands,seed)
        m.load_state_dict(ck['model'],strict=True);nets[name]=m.to(device).eval();weights[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),best_epoch=ck['best_epoch'])
    return nets,weights


def systems(cfg,nets,target):
    models=dict(nets);models['C2MS+P']=Projected(nets['C2MS'],target);models['L1S+P']=Projected(nets['L1S'],target)
    for v,spec in cfg['variants'].items():models['L1S+'+v]=ProjectedVariant(nets['L1S'],target,spec['dtype'],spec['tol'],spec['max_iter'])
    return models


def evaluate_target(cfg,sensor,target,seed,nets,dest,device='cuda'):
    models=systems(cfg,nets,target)
    rr=Path(cfg['data_root'])/data_files(target)['RR'];fr=Path(cfg['data_root'])/data_files(target)['FR'];want=cfg['data_sha256'][target]
    if sha(rr)!=want['RR'] or sha(fr)!=want['FR']:raise ValueError('test data mismatch '+target)
    wald=Wald(target,device);rows={name:[] for name in models}
    with h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for name,m in models.items():rows[name].append(dict(index=i,metrics=scene_metrics(m,a,b,wald,device)))
            print(json.dumps(dict(time_kst=now(),sensor=sensor,target=target,seed=seed,scene=i)),flush=True)
    means={name:{part:{k:float(np.mean([x['metrics'][part][k] for x in values])) for k in values[0]['metrics'][part]} for part in ('RR','FR')} for name,values in rows.items()}
    obs={name:float(np.mean([x['metrics']['OBS_RMSE_DN'] for x in values])) for name,values in rows.items()}
    iterations={name:dict(mean=float(np.mean(m.iterations)),max=int(max(m.iterations)),calls=len(m.iterations)) for name,m in models.items() if hasattr(m,'iterations')}
    judgement={name:judge(means[c],means[k]) for name,c,k in comparisons(cfg)}
    wins={f'{c}_vs_{k}':dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[c],rows[k])),
                             SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[c],rows[k]))) for _,c,k in comparisons(cfg)}
    timing=None
    if seed==cfg['latency_seed'] and target==sensor and device=='cuda':
        mv=SENSORS[target]['max_value']
        with h5py.File(rr) as r,h5py.File(fr) as f:
            xs={'RR_256':[torch.from_numpy((np.asarray(r[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')],
                'FR_512':[torch.from_numpy((np.asarray(f[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]}
        names=['C2MS','L1S','C2MS+P','L1S+P']+['L1S+'+v for v in cfg['variants']]
        timing={size:{name:latency(models[name],x,20 if '+' not in name else 10) for name in names} for size,x in xs.items()}
        timing['note']='GPU seconds per scene, median; scene 0; measured with no other GPU job'
    record=dict(target=target,training_sensor=sensor,transfer=target!=sensor,seed=seed,criteria='v2.4 (RR only)'+('; transfer test, reference only' if target!=sensor else ''),
                variants=cfg['variants'],timing=timing,projection_iterations=iterations,observation_rmse_DN=obs,rows=rows,means=means,judgement=judgement,scene_wins=wins,time_kst=now())
    write(dest/f'{target}_v2_4.json',record)
    if device=='cuda':
        result_images.render(models['C2MS'],models['L1S+P'],rr.parent,dest/target,names=('C2MS','C2MS_L1S+P'),seed=seed,scenes=cfg['scenes'],
            labels=('현재 기준 C2MS','C2MS (L1 + 분광 손실) + 사영'),rr_file=rr.name,scale=SENSORS[target]['max_value'],bands=SENSORS[target]['rgb'],
            land_cover=result_images.LAND_COVER if target=='QB' else {i:f'{target} 장면 {i}' for i in cfg['scenes']})
    return record


def evaluate(cfg,sensor,seed,device='cuda'):
    torch.manual_seed(seed);dest=Path(cfg['runtime'])/'runs'/run_name(sensor,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    nets,weights=load_models(cfg,sensor,seed,device)
    records={t:evaluate_target(cfg,sensor,t,seed,nets,dest,device) for t in targets(sensor)}
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=['C0','C2MS+P','L1S','L1S+P']+['L1S+'+v for v in cfg['variants']]
    fig,axes=plt.subplots(len(records),3,figsize=(16,3.6*len(records)),squeeze=False)
    for row,(t,rec) in zip(axes,records.items()):
        for ax,key in zip(row,('PSNR','SAM','ERGAS')):
            ax.bar(names,[rec['means'][n]['RR'][key]-rec['means']['C2MS']['RR'][key] for n in names],color=['#8d99ae','#adb5bd','#f4a261','#d1495b']+['#e9c46a']*len(cfg['variants']))
            ax.axhline(0,color='black',lw=.8);ax.set(title=f'{t} seed {seed} test RR {key} - C2MS'+(' (transfer, reference)' if t!=sensor else ''));ax.grid(axis='y',alpha=.3);ax.tick_params(axis='x',labelsize=7,rotation=30)
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=140);plt.close(fig)
    write(done,dict(time_kst=now(),weights=weights,targets={t:dict(judgement=r['judgement'],timing=r['timing'],transfer=r['transfer']) for t,r in records.items()},
                    scope='full 20-scene RR (judgement on the training sensor) and FR QNR (reference); WV2 transfer reference only; v2.4; per seed; no automatic baseline promotion'))


def summarize(cfg):
    """Per test target, means over the three seeds of the same-seed comparisons (v2.4, 0.04 dB); the reproduction of 79/80 is checked."""
    runtime=Path(cfg['runtime']);seeds=cfg['seeds'];out=dict(time_kst=now(),seeds=seeds,criteria='v2.4, mean of same-seed pairs; WV2 transfer reference only',targets={})
    for sensor in cfg['sensors']:
        for target in targets(sensor):
            per={s:json.loads((runtime/'runs'/run_name(sensor,s)/f'{target}_v2_4.json').read_text(encoding='utf8')) for s in seeds}
            names=list(per[seeds[0]]['means'])
            mean={n:{part:{k:float(np.mean([per[s]['means'][n][part][k] for s in seeds])) for k in per[seeds[0]]['means'][n][part]} for part in ('RR','FR')} for n in names}
            judgement={name:judge(mean[c],mean[k],len(seeds)) for name,c,k in comparisons(cfg)}
            per_seed={s:{name:{k:per[s]['judgement'][name]['delta_RR'][k] for k in ('PSNR','SAM','ERGAS')} for name,_,_ in comparisons(cfg)} for s in seeds}
            wins={name:{k:sum(per[s]['scene_wins'][name][k] for s in seeds) for k in ('PSNR','SAM')} for name in per[seeds[0]]['scene_wins']}
            keeps={v:all(abs(per_seed[s][f'variant_{v}_vs_L1S_P']['PSNR'])<=cfg['variant_keep_rule']['max_abs_psnr_db'] for s in seeds) and judgement[f'variant_{v}_vs_C2MS']['label']==judgement['official_L1S_P_vs_C2MS']['label'] for v in cfg['variants']}
            out['targets'][target]=dict(training_sensor=sensor,transfer=target!=sensor,means=mean,judgement=judgement,per_seed_delta_RR=per_seed,scene_wins=wins,variant_keeps_projection=keeps)
    out['reproduction']=reproduction(cfg,out)
    write(runtime/'summary.json',out);return out


def reproduction(cfg,out):
    """Largest |difference| of the 3-seed RR means of C2MS, L1S, C2MS+P and L1S+P against the 79 (QB) and 80 (GF2, WV3, WV2) summaries."""
    diffs={}
    for target,ref in cfg['reproduction_references'].items():
        if target not in out['targets']:continue
        r=json.loads((ROOT/ref['path']).read_text(encoding='utf8'));means=r['targets'][target]['means'] if 'targets' in r else r['means']
        diffs[target]={mine:max(abs(out['targets'][target]['means'][mine]['RR'][k]-means[theirs]['RR'][k]) for k in ('PSNR','SAM','ERGAS')) for mine,theirs in ref['names'].items()}
    return diffs
