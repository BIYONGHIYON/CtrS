"""89 stage B on a reduced c range (user decision 2026-11-20 20h: c >= 1.25 is not run because the curve already falls with c and time is
short; refine the small-c region 0 / 0.25 / 0.5 instead).

Runs after the sealed controller has stopped at the first c >= 1.25 launch (blocked by block_high_c.py) and the margin watcher has finished.
It continues only if stopped.json names nothing but that intended lock failure. Same trainer, sealed runtime config and command lines as
the controller (two jobs at once, evaluation first). Jobs: the evaluations the controller left (c = 1.0 reference), then
GF2/WV3/QB c = 0.25 and QB c = 0.75 (train + RR test evaluation, report only). Then the selection rule of config.json is applied unchanged
except for the reduced grid: QB grid 0-1.0 (0.25 step), centers 0.25-0.75, shared points 0 / 0.25 / 0.5 / 1.0, reference c = 1.0.
Writes runtime/selection_B_reduced.json and runtime/fine_c_complete.json (or fine_c_stopped.json)."""
import json,math,os,subprocess,sys,time,copy
from pathlib import Path
D89=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D89));sys.path.insert(1,str(D89.parents[2]))
os.chdir(D89)
import run as R
import model as M
import selection as S

LAM=1.2;SEED=42
BLOCKED=('C1.25','C1.50','C1.75','C2.00')
FINE=[('GF2',0.25),('QB',0.25),('WV3',0.25),('QB',0.75)]
GRID=[0.0,0.25,0.5,0.75,1.0];CENTERS=[0.25,0.5,0.75];SHARED=[0.0,0.25,0.5,1.0]


def log(**k):print(json.dumps(dict(time_kst=R.now(),**k),ensure_ascii=False),flush=True)


def wait_controller():
    chain=Path('C:/CtrS-p89/chain.log');watch=Path('C:/CtrS-p89/watch_margin.log')
    while True:
        t=chain.read_text(encoding='utf8',errors='ignore') if chain.exists() else '';w=watch.read_text(encoding='utf8',errors='ignore') if watch.exists() else ''
        if '89 run exit' in t and ('continue_without_margin exit' in w or 'nothing started' in w):return t.split('89 run exit')[-1].strip().splitlines()[0]
        time.sleep(60)


def main():
    exit_line=wait_controller();log(controller_exit=exit_line)
    conf=R.load_conf();root=Path(conf['runtime']);cp=root/'runtime_config.json';cfg=json.loads(cp.read_text(encoding='utf8'))
    guard=R.lock(root/'controller.lock')
    stopped=root/'stopped.json'
    reason=json.loads(stopped.read_text(encoding='utf8'))['reason'] if stopped.exists() else ''
    if not reason.startswith('train failed') or not any(b in reason for b in BLOCKED):
        R.write(root/'fine_c_stopped.json',dict(time_kst=R.now(),reason='controller stop is not the intended c >= 1.25 lock failure: '+repr(reason)));log(refused=reason);sys.exit(4)
    R.write(root/'fine_c_started.json',dict(time_kst=R.now(),decision='user 2026-11-20 20h: skip c >= 1.25, refine small c (0 / 0.25 / 0.5)',controller_stop=reason,
            script=str(Path(__file__).resolve()),script_sha256=R.sha(Path(__file__).resolve()),grid_QB=GRID,centers=CENTERS,shared=SHARED))
    keys_ref=[(s,M.screen_arm(LAM,1.0,'B'),SEED) for s in ('GF2','WV3','QB')]
    keys_fine=[(s,M.screen_arm(LAM,c,'B'),SEED) for s,c in FINE]
    out=lambda k:root/'runs'/R.run_name(*k)
    trained=lambda k:(out(k)/'training_complete.json').exists()
    evaluated=lambda k:(out(k)/'evaluation_complete.json').exists()
    for k in keys_ref:
        if not trained(k):
            R.write(root/'fine_c_stopped.json',dict(time_kst=R.now(),reason='reference run not trained: '+R.run_name(*k)));sys.exit(5)
    jobs=[('evaluate',k) for k in keys_ref if not evaluated(k)]+[('train',k) for k in keys_fine if not trained(k)]+[('evaluate',k) for k in keys_fine if trained(k) and not evaluated(k)]
    running={};problem=None
    def launch(kind,k):
        o=out(k);o.mkdir(parents=True,exist_ok=True);f=(o/(kind+'.log')).open('a',encoding='utf8')
        args=[sys.executable,'-u',str(D89/'train.py'),'--config',str(cp),'--mode',kind,'--sensor',k[0],'--arm',k[1],'--seed',str(k[2])]
        if kind=='train' and (o/'checkpoints/initial.pt').exists():args+=['--resume']
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT);log(launched=kind,run=R.run_name(*k),pid=p.pid);return p,f
    while True:
        if problem is None:
            for j in sorted(jobs,key=lambda j:j[0]!='evaluate'):
                if len(running)>=cfg['max_parallel']:break
                jobs.remove(j);running[j]=launch(*j)
        time.sleep(10)
        for j,(p,f) in list(running.items()):
            if p.poll() is None:continue
            f.close();del running[j];kind,k=j
            ok=p.returncode==0 and (trained(k) if kind=='train' else evaluated(k))
            if ok and kind=='train':
                v=R.validation_of(root,k)
                if not all(math.isfinite(x) for x in (v['psnr'],v['sam'],v['mse'])) or v['epoch']<1:ok=False
                else:jobs.append(('evaluate',k))
            log(finished=kind,run=R.run_name(*k),ok=ok,exit=p.returncode)
            if not ok and problem is None:problem=f'{kind} failed {R.run_name(*k)} (exit {p.returncode})'
        if not running and (problem or not jobs):break
    if problem:
        R.write(root/'fine_c_stopped.json',dict(time_kst=R.now(),reason=problem,note='no new job after the problem; running jobs finished'));log(stopped=problem);sys.exit(3)
    val={s:{} for s in ('QB','GF2','WV3')}
    for s in val:
        for c in GRID:
            k=(s,M.screen_arm(LAM,c,'B'),SEED)
            if trained(k):val[s][c]=R.validation_of(root,k)
    red=copy.deepcopy(cfg);red['stage_B']['grid']={'QB':GRID};red['stage_B']['center_candidates']=CENTERS;red['stage_B']['shared_points']=SHARED
    sel=S.choose('B',red,val);sel.update(time_kst=R.now(),lambda_star=LAM,reduced_range='c >= 1.25 not run (user decision 2026-11-20 20h); grid 0-1.0',
        rerun_spread={s:dict(stage_A=R.validation_of(root,(s,M.screen_arm(LAM,1.0),SEED)),stage_B=val[s][1.0]) for s in val})
    R.write(root/'selection_B_reduced.json',sel)
    R.write(root/'fine_c_complete.json',dict(time_kst=R.now(),lambda_star=LAM,c_star=sel['chosen'],at_boundary=sel['at_boundary'],next='confirmation after the user decision'))
    log(c_star=sel['chosen'],at_boundary=sel['at_boundary'])


if __name__=='__main__':main()
