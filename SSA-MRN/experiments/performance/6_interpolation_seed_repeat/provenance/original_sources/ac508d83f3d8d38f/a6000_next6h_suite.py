"""Detached staged GF2 repeat/QB input queue with a timing-only six-hour gate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import a6000_followup_suite as common

ROOT = common.ROOT
PLAN = ROOT/'scripts/a6000_next_6h_plan.json'
RUN = ROOT/'experiments/a6000_gf2_repeat_qb_input'
common.PLAN, common.RUN = PLAN, RUN


def inventory(plan):
    result = common.inventory(plan)
    for p in [Path(__file__), PLAN, ROOT/'scripts/run_a6000_next6h.sh',ROOT/'scripts/evaluate_a6000_next6h.py',ROOT/'scripts/verify_a6000_next6h_report.py']:
        result['source'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result['reference_manifest_sha256'] = hashlib.sha256((Path(plan['reference_root'])/'reference_manifest.json').read_bytes()).hexdigest()
    return result


def can_start_stage2(elapsed, policy):
    return elapsed+policy['stage2_training_reserve_seconds']+policy['evaluation_reserve_seconds']<=policy['target_seconds']


def estimate_stage1(active, elapsed, plan):
    remaining=[]
    for name in active:
        p=RUN/name/'history.jsonl'
        if not p.exists():return None
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        if len(rows)<10:return None
        remaining.append((plan['training']['epochs']-rows[-1]['epoch'])*statistics.median(x['seconds'] for x in rows[-5:]))
    if not remaining:return None
    return {'elapsed_seconds':elapsed,'estimated_stage1_remaining_seconds':max(remaining),
            'stage2_estimated_feasible':can_start_stage2(elapsed+max(remaining),plan['budget_policy']),
            'method':'slowest active run, median of latest5 actual completed epochs; timing only, advisory'}


def verify_queue(plan):
    assert plan['parallel']==4
    assert len(plan['runs'])==6 and [r['stage'] for r in plan['runs']]==[1,1,1,1,2,2]
    assert all(r['seed'] in [43,44] and r['k']==6 for r in plan['runs'])
    assert {r['variant'] for r in plan['runs'] if r['stage']==1}=={'baseline','interp23'}
    assert {r['variant'] for r in plan['runs'] if r['stage']==2}=={'interp23_input'}
    for available in [True,False]:
        pending=list(plan['runs']); active=[]; events=[]; tick=0; peak=0; stage=1
        while pending or active:
            active=[(row,end) for row,end in active if end>tick]
            if not active and pending:
                stage=pending[0]['stage']
                if stage==2 and not available:pending=[];break
            while pending and pending[0]['stage']==stage and len(active)<4:
                row=pending.pop(0); active.append((row,tick+3+row['seed']%2)); events.append((stage,tick));peak=max(peak,len(active))
            assert len({row['stage'] for row,_ in active})<=1
            tick+=1
        assert peak==4 and sum(s==1 for s,t in events)==4 and sum(s==2 for s,t in events)==(2 if available else 0)
        if available:assert min(t for s,t in events if s==2)>=max(t+3+plan['runs'][i]['seed']%2 for i,(s,t) in enumerate(events[:4]))
    p=plan['budget_policy']; boundary=p['target_seconds']-p['stage2_training_reserve_seconds']-p['evaluation_reserve_seconds']
    assert can_start_stage2(boundary,p) and not can_start_stage2(boundary+1,p)
    print('PASS:4/2 staged queue, max4, no stage overlap, whole-pair deferral and time gate boundary')


def worker(plan):
    lock=(RUN/'queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=json.loads((RUN/'inventory.json').read_text()); pending=list(plan['runs']); active={};deferred=[];stage=1
    started=json.loads((RUN/'state.json').read_text()).get('started_at',time.time())
    try:
        common.write(RUN/'references.json',plan['references'])
        while pending or active:
            if inventory(plan)!=expected:raise RuntimeError('Source/data/reference changed; preserve checkpoints and halt this queue')
            for name,(child,log) in list(active.items()):
                code=child.poll()
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} failed with exit{code}')
            if not active and pending:
                stage=pending[0]['stage']
                if stage==2 and not can_start_stage2(time.time()-started,plan['budget_policy']):
                    # Completed stage2 runs may be reused after an explicit resume.
                    deferred=[r for r in pending if not (RUN/common.name(r)/'complete.json').exists()]
                    pending=[r for r in pending if (RUN/common.name(r)/'complete.json').exists()]
            while pending and pending[0]['stage']==stage and len(active)<plan['parallel']:
                if common.quota_free()<256*1024**2:raise RuntimeError('Checkpoint quota reserve exhausted')
                row=pending.pop(0);name=common.name(row);directory=RUN/name;directory.mkdir(exist_ok=True)
                c=common.config(plan,row);cfg=directory/'config.json'
                if cfg.exists() and json.loads(cfg.read_text())!=c:raise RuntimeError('Existing config mismatch')
                common.write(cfg,c);done=directory/'complete.json'
                if done.exists():
                    assert json.loads(done.read_text())['config']==c;continue
                cmd=[sys.executable,str(ROOT/'scripts/train_a6000_followup.py'),'--config',str(cfg),'--output',str(directory)]
                if (directory/'latest.pt').exists():cmd+=['--resume','latest']
                log=(directory/'train.log').open('a');env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                active[name]=(child,log)
            state={'pid':os.getpid(),'status':'running','started_at':started,'active':{k:v[0].pid for k,v in active.items()},'pending':len(pending),'stage':stage,'heartbeat':time.time(),'deferred':deferred}
            if stage==1:state['timing_projection']=estimate_stage1(active,time.time()-started,plan)
            common.write(RUN/'state.json',state)
            if active:time.sleep(5)
        scores={common.name(row):json.loads((RUN/common.name(row)/'complete.json').read_text())['best_mse'] for row in plan['runs'] if (RUN/common.name(row)/'complete.json').exists()}
        scores.update({common.name(row):json.loads((Path(row['path'])/'complete.json').read_text())['best_mse'] for row in plan['references']})
        common.write(RUN/'validation_scores.json',scores)
        common.write(RUN/'state.json',{'pid':os.getpid(),'status':'evaluating','started_at':started,'active':{},'deferred':deferred,'heartbeat':time.time()})
        out=ROOT/'docs/assets/a6000_gf2_repeat_qb_input'
        with (RUN/'evaluation.log').open('a') as log:
            evaluation=subprocess.Popen([sys.executable,str(ROOT/'scripts/evaluate_a6000_next6h.py'),'--data-root','/home/gpu_04/CtrS-a6000/SSA-MRN/data/dataset','--run-root',str(RUN),'--output',str(out)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while evaluation.poll() is None:
                common.write(RUN/'state.json',{'pid':os.getpid(),'status':'evaluating','started_at':started,'active':{'evaluation':evaluation.pid},'deferred':deferred,'heartbeat':time.time()});time.sleep(5)
            if evaluation.returncode:raise RuntimeError('Evaluation failed; see evaluation.log; training checkpoints retained')
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_a6000_next6h_report.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        common.write(RUN/'state.json',{'pid':os.getpid(),'status':'finished_with_deferred' if deferred else 'finished','started_at':started,'finished_at':time.time(),'active':{},'deferred':deferred,'completed_new_runs':sum((RUN/common.name(r)/'complete.json').exists() for r in plan['runs']),'evaluation_verified':True,'heartbeat':time.time()})
    except BaseException as e:
        children=[v[0] for v in active.values()]
        if 'evaluation' in locals() and evaluation.poll() is None:children.append(evaluation)
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for child,log in active.values():log.close()
        common.write(RUN/'state.json',{'pid':os.getpid(),'status':'failed','started_at':started,'active':{},'error':str(e),'heartbeat':time.time()})
        raise



def verify_hardware(plan):
    common.check(plan)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU in use; skip hardware verification')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_a6000_followup.py')],check=True)
    # Management process starts these short real-data verification children; not research results.
    directory=ROOT/'experiments/next6h_smoke';directory.mkdir(parents=True,exist_ok=True);children=[]
    try:
        for row in plan['runs'][:4]:
            c=common.config(plan,row);c.update(epochs=2,limit=32);d=directory/common.name(row);d.mkdir(exist_ok=True)
            cfg=d/'config.json';common.write(cfg,c);log=(d/'train.log').open('w')
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/train_a6000_followup.py'),'--config',str(cfg),'--output',str(d)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            children.append((child,log,d))
        peak=0
        while any(child.poll() is None for child,log,d in children):
            peak=max(peak,sum(child.poll() is None for child,log,d in children));time.sleep(.2)
        for child,log,d in children:
            log.close();assert child.returncode==0,d
            done=json.loads((d/'complete.json').read_text());assert done['epoch']==2
            rows=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()];assert len(rows)==2
            assert all(__import__('math').isfinite(x[key]) for x in rows for key in ['train_mse','val_mse','val_psnr_band_mean_peak1','val_sam_deg'])
        assert peak==4
        common.write(ROOT/'docs/operations/a6000_next6h_hardware_verification.json',{'status':'passed','parallel_children_peak':peak,'sensor':'GF2','batch_size':32,'micro_batch':32,'epochs':2,'train_and_val_limit':32,'scope':'verification only; not research training results','completed_at':time.time()})
        print('PASS:4 actual GF2 CUDA children, batch=micro32,2 verification epochs, checkpoints and finite validation MSE/PSNR/SAM')
    finally:
        for child,log,d in children:
            if child.poll() is None:child.terminate();child.wait(timeout=30)
            if not log.closed:log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','verify-queue','verify-hardware','start','resume-latest','status','logs','worker']);a=p.parse_args()
    plan=json.loads(PLAN.read_text());RUN.mkdir(parents=True,exist_ok=True)
    state=json.loads((RUN/'state.json').read_text()) if (RUN/'state.json').exists() else {'status':'not_started'}
    if a.command=='verify-hardware':verify_hardware(plan)
    elif a.command=='verify-queue':verify_queue(plan)
    elif a.command=='check':
        print(json.dumps(common.check(plan),indent=2));print('inventory files',len(inventory(plan)['source']))
    elif a.command in ['start','resume-latest']:
        if common.alive(state.get('pid')) or any(common.alive(pid) for pid in state.get('active',{}).values()):raise RuntimeError('Controller or child still running')
        if a.command=='start' and any(RUN.glob('*/latest.pt')):raise RuntimeError('Use resume-latest for existing checkpoints')
        common.check(plan);verify_queue(plan)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU already in use')
        for name,value in [('plan.json',plan),('inventory.json',inventory(plan))]:
            path=RUN/name
            if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Plan/source/reference mismatch; use a new run directory')
            common.write(path,value)
        started=state.get('started_at',time.time())
        # Persist start budget before spawning worker; child startup cannot read a stale state.
        common.write(RUN/'state.json',{'status':'starting','started_at':started,'active':{}})
        with (RUN/'controller.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        current=json.loads((RUN/'state.json').read_text())
        if current['status']=='starting':common.write(RUN/'state.json',dict(current,pid=child.pid))
        print('controller PID=',child.pid)
    elif a.command=='worker':worker(plan)
    elif a.command=='status':print(json.dumps(dict(state,controller_alive=common.alive(state.get('pid'))),indent=2))
    else:
        for path in sorted(RUN.glob('*/train.log')):print(path.parent.name);print('\n'.join(path.read_text().splitlines()[-3:]))
        if (RUN/'evaluation.log').exists():print('evaluation');print('\n'.join((RUN/'evaluation.log').read_text().splitlines()[-3:]))

if __name__=='__main__':main()
