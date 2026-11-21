"""Stop 89 stage B before the c >= 1.25 runs (user decision 2026-11-20 20h: the curve already falls with c, skip c >= 1.25 for time).
The sealed controller is not touched. This process holds the worker lock (58 trainer msvcrt lock on runs/<run>/worker.lock) of every
stage-B run with c in 1.25-2.0 at lambda 1.2, so the first such launch exits at once with a lock error before any checkpoint; the
controller then launches nothing new, lets the running trainings finish and exits (stop rule). The remaining small-c runs are trained
by run_fine_c.py. Record: runtime/stage_B_high_c_blocked.json. Exits by itself when chain.log shows the controller exit and the
margin watcher has finished."""
import json,msvcrt,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
root=Path('C:/CtrS-p89/run');now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat();held=[];names=[]
for c in (1.25,1.5,1.75,2.0):
    for s in ('GF2','WV3','QB'):
        d=root/'runs'/f'{s}_BL1.20C{c:.2f}_s42'
        if (d/'checkpoints').exists():raise RuntimeError('already initialized: '+str(d))
        d.mkdir(parents=True,exist_ok=True);f=(d/'worker.lock').open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);held.append(f);names.append(d.name)
rec=dict(time_kst=now(),decision='user 2026-11-20 20h: stage B c >= 1.25 is not run (the curve already falls with c; time); small c refined instead',
         locks=len(held),runs=names,mechanism='worker locks held; the first c >= 1.25 launch fails before any checkpoint and the controller stops')
(root/'stage_B_high_c_blocked.json').write_text(json.dumps(rec,indent=1),encoding='utf8');print('HOLDING',len(held),'locks',flush=True)
log=Path('C:/CtrS-p89/chain.log');wlog=Path('C:/CtrS-p89/watch_margin.log')
while True:
    t=log.read_text(encoding='utf8',errors='ignore') if log.exists() else '';w=wlog.read_text(encoding='utf8',errors='ignore') if wlog.exists() else ''
    if '89 run exit' in t and ('continue_without_margin exit' in w or 'nothing started' in w):break
    time.sleep(60)
rec['released_kst']=now();(root/'stage_B_high_c_blocked.json').write_text(json.dumps(rec,indent=1),encoding='utf8');print('RELEASED',flush=True)
