"""Evaluate each validation-selected run on the 20 RR/FR test scenes of its sensor (WV3 runs also on WV2, reference only).

Systems (same seed and sensor): the run raw and with the K10 projection (`+P`); the control C2MS-SP = 65 C2MS_L1S (150-epoch cosine) + P and
65 raw, cached per (sensor, seed, target) in runtime/controls/. On WV2 (a sensor the weights were not trained on) the C2MS-SP definition
adds the band calibration of experiment 82 before the projection (`+CP`); raw and +P are reported as well. RR (v2.4 judgement) and FR
QNR/D_lambda/D_s (reference). summarize() picks the training length per sensor by validation only (smallest best validation MSE over the
branches) and judges that branch + P against C2MS-SP on the test; every other branch is reported as the length curve."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
from model import M54,ARMS,BRANCHES
from prepare import Wald,SENSORS,data_files,Projected
from train import T,sha,write,now,run_name,branch_start
from common.metrics import rr_metrics,fr_metrics
spec=importlib.util.spec_from_file_location('c2ms54_eval',HERE.parent/'54_multi_sensor_scale_mixed_training/evaluate.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
spec=importlib.util.spec_from_file_location('exp82_prepare',HERE.parent/'82_cross_sensor_band_calibration/prepare.py')
P82=importlib.util.module_from_spec(spec);spec.loader.exec_module(P82)
NOISE={1:0.06}  # docs/judgement_criteria.md v2.4: single seed 0.06 dB, two or more seeds 0.04 dB
TRANSFER={'WV3':['WV2']}


class Calibrated(P82.Calibrated):
    """82 calibration ('scene') with its reference ratios moved to the input device (82 ran on the CPU)."""
    def forward(self,pan,lms,ms):
        self.r_train=self.r_train.to(pan.device);return super().forward(pan,lms,ms)


def targets(sensor):return [sensor]+TRANSFER.get(sensor,[])


def judge(cand,ctrl,seeds=1):
    d={k:cand['RR'][k]-ctrl['RR'][k] for k in cand['RR']};threshold=NOISE.get(seeds,0.04)
    parts={'PSNR':d['PSNR']>0,'SAM':d['SAM']<0,'ERGAS':d['ERGAS']<0}
    label='개선' if all(parts.values()) and d['PSNR']>=threshold else '부분 개선' if any(parts.values()) else '개선 없음'
    return dict(delta_RR=d,delta_FR_reference={k:cand['FR'][k]-ctrl['FR'][k] for k in cand['FR']},rr_components_improved=parts,noise_threshold=threshold,above_noise=d['PSNR']>=threshold,label=label)


def systems(net,name,sensor,target,cfg):
    out={name:net,name+'+P':Projected(net,target)}
    if target!=sensor:
        ref=json.loads((HERE.parent/cfg['calibration_reference'][sensor]).read_text(encoding='utf8'))
        out[name+'+CP']=Projected(Calibrated(net,ref,'scene'),target)
    return out


def official(name,sensor,target):
    """(candidate, control) that define the judgement: C2MS-SP on the training sensor is 65 + P, on an unseen sensor 65 + calibration + P."""
    s='+P' if target==sensor else '+CP';return name+s,'L1S'+s


def run_systems(cfg,target,models):
    files=data_files(target);rr=Path(cfg['data_root'])/files['RR'];fr=Path(cfg['data_root'])/files['FR']
    want=(cfg['data_sha256'] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'])[target]
    if sha(rr)!=want['RR'] or sha(fr)!=want['FR']:raise ValueError('test data mismatch '+target)
    s=SENSORS[target]['max_value'];rows={n:[] for n in models}
    with torch.inference_mode(),h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected all 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for n,m in models.items():
                rows[n].append(dict(index=i,metrics=dict(RR=rr_metrics(a['gt'],V.predict(m,a['pan'],a['lms'],a['ms'],s,'cuda')),
                                                         FR=fr_metrics(V.predict(m,b['pan'],b['lms'],b['ms'],s,'cuda'),b['lms'],b['ms'],b['pan']))))
            print(json.dumps(dict(time_kst=now(),target=target,scene=i,systems=list(models))),flush=True)
    return rows


def load(path,bands,seed):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=M54.Model('C2MS',bands,seed);m.load_state_dict(ck['model'],strict=True)
    return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'),metrics=ck.get('metrics'))


def control_rows(cfg,sensor,target,seed):
    p=Path(cfg['runtime'])/'controls'/f'{sensor}_s{seed}_{target}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    net,w=load(HERE.parents[2]/cfg['control'][sensor].replace('<seed>',str(seed)),SENSORS[sensor]['bands'],seed)
    rows=run_systems(cfg,target,systems(net,'L1S',sensor,target,cfg));rec=dict(rows=rows,weights=w,time_kst=now());write(p,rec);return rec


def means(rows):return {n:{part:{k:float(np.mean([x['metrics'][part][k] for x in v])) for k in v[0]['metrics'][part]} for part in ('RR','FR')} for n,v in rows.items()}


def evaluate(cfg,sensor,arm,seed):
    T.setup(seed);dest=Path(cfg['runtime'])/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    net,w=load(dest/'checkpoints/best.pt',SENSORS[sensor]['bands'],seed);records={}
    for target in targets(sensor):
        ctrl=control_rows(cfg,sensor,target,seed);rows=run_systems(cfg,target,systems(net,arm,sensor,target,cfg));rows.update(ctrl['rows']);m=means(rows)
        cand,base=official(arm,sensor,target)
        judgement={'official_vs_C2MS_SP':judge(m[cand],m[base]),'raw_vs_65_raw':judge(m[arm],m['L1S'])}
        if target!=sensor:judgement['P_vs_65_P']=judge(m[arm+'+P'],m['L1S+P'])
        wins=dict(PSNR=sum(x['metrics']['RR']['PSNR']>y['metrics']['RR']['PSNR'] for x,y in zip(rows[cand],rows[base])),SAM=sum(x['metrics']['RR']['SAM']<y['metrics']['RR']['SAM'] for x,y in zip(rows[cand],rows[base])))
        records[target]=dict(target=target,training_sensor=sensor,transfer=target!=sensor,arm=arm,seed=seed,official=[cand,base],
                             criteria='v2.4 (RR only)'+('; transfer test, reference only' if target!=sensor else ''),rows=rows,means=m,judgement=judgement,scene_wins_official=wins,time_kst=now())
        write(dest/'evaluation'/f'{target}_v2_4.json',records[target])
    tc=json.loads((dest/'training_complete.json').read_text(encoding='utf8'));v65=ctrl['weights']['metrics']
    validation=dict(run=tc['best_val'],best_epoch=tc['best_epoch'],control_65=v65,delta_psnr_vs_65=tc['best_val']['val_psnr']-v65['val_psnr'],
                    delta_sam_vs_65=tc['best_val']['val_sam']-v65['val_sam'],delta_mse_vs_65=tc['best_val']['val_mse']-v65['val_mse'])
    write(done,dict(time_kst=now(),weights=w,epochs=int(cfg['epochs'][arm]),branch=dict(zip(('trunk_epoch','decay_epochs'),branch_start(cfg,arm))) if arm in BRANCHES else None,
                    validation=validation,targets={t:dict(official=r['official'],judgement=r['judgement'],means={k:r['means'][k] for k in r['official']},transfer=r['transfer']) for t,r in records.items()},
                    scope='full 20-scene RR (judgement on the training sensor) and FR QNR (reference); WV2 reference only; v2.4; single seed; K10 projection; no automatic baseline promotion'))


def summarize(cfg,seed=42):
    """Per sensor: the length curve (validation and test of every branch and the trunk) and the validation-chosen length judged on the test."""
    runtime=Path(cfg['runtime']);out=dict(time_kst=now(),seed=seed,criteria='v2.4, single seed (0.06 dB); WV2 reference only',
                                          choice_rule=cfg['choice_rule'],sensors={})
    for sensor in cfg['sensors']:
        runs={a:json.loads((runtime/'runs'/run_name(sensor,a,seed)/'evaluation_complete.json').read_text(encoding='utf8')) for a in ARMS
              if (runtime/'runs'/run_name(sensor,a,seed)/'evaluation_complete.json').exists()}
        branches=[a for a in BRANCHES if a in runs]
        if len(branches)<len(BRANCHES):out['sensors'][sensor]=dict(incomplete=sorted(set(BRANCHES)-set(branches)));continue
        chosen=min(branches,key=lambda a:(runs[a]['validation']['run']['val_mse'],BRANCHES[a]))
        curve={a:dict(epochs=r['epochs'],best_epoch=r['validation']['best_epoch'],validation=r['validation']['run'],delta_val_psnr_vs_65=r['validation']['delta_psnr_vs_65'],
                      test={t:dict(official=x['official'],judgement=x['judgement']['official_vs_C2MS_SP']['label'],delta_RR=x['judgement']['official_vs_C2MS_SP']['delta_RR'],
                                   means=x['means']) for t,x in r['targets'].items()}) for a,r in runs.items()}
        out['sensors'][sensor]=dict(chosen_by_validation=chosen,chosen_epochs=BRANCHES[chosen],
                                    judgement=runs[chosen]['targets'][sensor]['judgement']['official_vs_C2MS_SP'],
                                    schedule_check_B150_vs_65=runs['B150']['targets'][sensor]['judgement']['raw_vs_65_raw'],curve=curve)
    write(runtime/'summary.json',out);return out
