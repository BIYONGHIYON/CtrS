"""Recompute v2.3 labels from retained paired measurements; never train or select on test."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def judge(a,b,n):
    d={s:{k:b[s][k]-a[s][k] for k in a[s]} for s in a};t=.06 if n==1 else .04
    ratio=b['FR']['OBS_RMSE_DN']/a['FR']['OBS_RMSE_DN']
    components={'PSNR':d['RR']['PSNR']>0,'SAM':d['RR']['SAM']<0,'ERGAS':d['RR']['ERGAS']<0}
    guards={'RR2':d['RR2']['PSNR']>=-.05 and d['RR2']['SAM']<=.06,
            'FR_WALD':d['FR_WALD']['PSNR']>=-.07 and d['FR_WALD']['SAM']<=.04,'FR_OBS':ratio<=1.1}
    primary=all(components.values()) and d['RR']['PSNR']>=t
    return {'criteria_version':'v2.3','n_seeds':n,'delta':d,'RR_components_improved':components,'PSNR_noise_threshold_dB':t,
            'PSNR_noise_threshold_pass':d['RR']['PSNR']>=t,'primary':primary,'guards':guards,'observation_ratio':ratio,
            'label':'개선' if primary and all(guards.values()) else ('부분 개선' if any(components.values()) else '개선 없음'),
            'QNR_used_for_decision':False,'comparison':'same-run W2 C2 control; not current F1'}
def calculate(root=ROOT):
    from statistics import mean
    seeds=(42,43,44);values={};per_seed={}
    for seed in seeds:
        arms=[]
        for arm in ('baseline','hf_gate'):
            p=root/f'results/seed_{seed}/{arm}';proof=read(p/'evaluation_complete.json')
            ck=root/f'checkpoints/{arm}_QB_seed{seed}/best.pt';r=p/'evaluations/best/results.json'
            assert proof['best_sha256']==sha(ck) and proof['results_sha256']['best']==sha(r)
            result=read(r);assert all(v['count']==20 for v in result['splits'].values())
            arms.append({s:v['mean'] for s,v in result['splits'].items()})
        values[seed]=arms;per_seed[str(seed)]=judge(*arms,1)
    aggregate=[{s:{k:mean(values[seed][arm][s][k] for seed in seeds) for k in values[42][arm][s]} for s in values[42][arm]} for arm in (0,1)]
    return {'criteria_version':'v2.3','seeds':list(seeds),'original_judgement_preserved':'provenance/judgement_v2_2.json',
            'criteria_sha256':sha(root/'provenance/current_judgement_criteria.md'),
            'aggregation':'all paired seed arithmetic means; OBS ratio of seed means; no majority veto',
            'measurement_scope':'same saved best measurements; no new inference, training or test selection',
            'per_seed':per_seed,'seed_means':dict(zip(('baseline','hf_gate'),aggregate)),'aggregate':judge(*aggregate,3)}
if __name__=='__main__':
    result=calculate();target=ROOT/'results/judgement_v2_3.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(result['aggregate']['label'])
