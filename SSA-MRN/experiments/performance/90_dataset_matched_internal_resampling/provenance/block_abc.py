"""Stop the sealed 90 controller after the N variant (user decision 2026-11-21 02h: choose A/B/C on 50-epoch screening runs, as the 89
lambda search, instead of 150-epoch K2D runs). Holds the worker lock (58 trainer msvcrt lock on runs/<run>/worker.lock) of the A/B/C
trunks (AT, BT, CT on GF2, QB, WV3), so the controller's first A/B/C launch exits at once before any checkpoint and the controller stops
(no new job; its running N job finishes). run.py --mode screen then finishes N and runs the screening. Record:
runtime/abc_blocked.json. Exits by itself when chain.log shows the controller exit."""
import json,msvcrt,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
root=Path('C:/CtrS-p90/run');now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat();held=[];names=[]
for v in 'ABC':
    for s in ('GF2','QB','WV3'):
        d=root/'runs'/f'{s}_{v}T_s42'
        if (d/'checkpoints').exists():raise RuntimeError('already initialized: '+str(d))
        d.mkdir(parents=True,exist_ok=True);f=(d/'worker.lock').open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);held.append(f);names.append(d.name)
rec=dict(time_kst=now(),decision='user 2026-11-21 02h: A/B/C chosen on 50-epoch screening runs; the 150-epoch A/B/C runs are not trained',locks=len(held),runs=names,
         mechanism='worker locks held; the first A/B/C trunk launch fails before any checkpoint and the controller stops after its running N job')
(root/'abc_blocked.json').write_text(json.dumps(rec,indent=1),encoding='utf8');print('HOLDING',len(held),'locks',flush=True)
log=Path('C:/CtrS-p90/chain.log')
while '90 run exit' not in (log.read_text(encoding='utf8',errors='ignore') if log.exists() else ''):time.sleep(60)
rec['released_kst']=now();(root/'abc_blocked.json').write_text(json.dumps(rec,indent=1),encoding='utf8');print('RELEASED',flush=True)
