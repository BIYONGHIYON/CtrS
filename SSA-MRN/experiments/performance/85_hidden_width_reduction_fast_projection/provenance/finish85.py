"""Finish experiment 85 after the WV3 FT0L evaluation failed (06:21) on the WV2 transfer path: the 82 calibration ratios stayed on the CPU.
Only evaluate.py changed (Calibrated moves its ratios to the input device, as 84). The sealed runtime config is not modified; both training
runs were completed with the sealed code. Steps: record the fix, evaluate WV3 FT0L, the validation gate, the final latency, all_complete."""
import os,sys,json,subprocess
from pathlib import Path
D='C:/CtrS-p85/SSA-MRN/experiments/performance/85_hidden_width_reduction_fast_projection';sys.path.insert(0,D);os.chdir(D)
import run as R
from train import sha,write,now
root,cp,cfg=R.runtime_cfg();key='experiments/performance/85_hidden_width_reduction_fast_projection/evaluate.py'
now_hashes=R.source_hashes();changed={k:dict(sealed=cfg['source_hashes'].get(k),now=v) for k,v in now_hashes.items() if cfg['source_hashes'].get(k)!=v}
if set(changed)!={key}:raise RuntimeError('unexpected source change: '+json.dumps(changed))
for s in ('QB','WV3'):
    if not (root/'runs'/f'{s}_FT0L_s42'/'training_complete.json').exists():raise RuntimeError('training incomplete '+s)
write(root/'evaluation_fix.json',dict(time_kst=now(),failure='WV3 FT0L evaluation, WV2 transfer: RuntimeError cuda:0 and cpu (82 Calibrated r_train on the CPU)',
      fix='evaluate.Calibrated moves r_train to the input device (as 84 evaluate.py); no other source change',changed=changed,
      scope='WV3 FT0L evaluation, validation gate, final latency; training runs and the QB FT0L evaluation used the sealed code; runtime_config.json unchanged'))
log=open(root/'runs'/'WV3_FT0L_s42'/'evaluate_fixed.log','w',encoding='utf8')
subprocess.run([sys.executable,'-u','train.py','--config',str(cp),'--mode','evaluate','--sensor','WV3','--arm','FT0L','--seed','42'],stdout=log,stderr=subprocess.STDOUT,check=True);log.close()
from evaluate import stage_gate
gate=stage_gate(cfg);write(root/'stage_gate.json',gate);print('GATE',json.dumps(gate['passed']),gate['chosen'],flush=True)
queue=[(s,a,cfg['latency_seed']) for s in cfg['stage1_sensors'] for a in cfg['trained_arms']]
R.snapshot(cfg,[],'final timing',queue);R.final_measure(cfg,gate['chosen'])
R.snapshot(cfg,[],'complete',queue);write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(queue),stage_gate=gate['chosen'],note='finished by finish85.py after the evaluation fix (evaluation_fix.json)'))
print('FINISH85 DONE',flush=True)
