"""Resume sealed suite with the missing module search path supplied externally."""
import os,sys,json,hashlib,subprocess,runpy
from pathlib import Path
from datetime import datetime
BASE=Path('C:/CtrS-p70-71');ROOT=BASE/'SSA-MRN';PERF=ROOT/'experiments/performance';OUT=BASE/'recovery_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
    OUT.mkdir(exist_ok=True);r=BASE/'run70';cfg=json.loads((r/'runtime_config.json').read_text(encoding='utf-8'));before={}
    for sensor,arm,seed in cfg['queue']:
        p=r/'runs'/f'{sensor}_{arm}_s{seed}';done=json.loads((p/'training_complete.json').read_text(encoding='utf-8'))
        assert done['epoch']==150 and done['fingerprint']==cfg['fingerprint']
        assert sha(p/'checkpoints/best.pt')==done['best_sha256']
        before[str(p)]=dict(epoch=done['epoch'],best_sha256=done['best_sha256'],history_sha256=sha(p/'history.json'),completion_sha256=sha(p/'training_complete.json'))
    for name,expected in cfg['source_hashes'].items():assert sha(ROOT/name)==expected, name
    os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['PYTHONUTF8']='1';os.environ['PYTHONIOENCODING']='utf-8'
    for n in ('70_c2ms_dilated_backbone','71_coarse_pair_pan_blur'):
        p=PERF/n
        subprocess.run([sys.executable,'-X','utf8','-c',"import validation,common.interpolation; print('IMPORT PASS',validation.__file__,common.interpolation.__file__)"],cwd=p,check=True)
    record=dict(time=datetime.now().isoformat(),repair='PYTHONPATH points to SSA-MRN; sealed training sources and checkpoints unchanged',before=before)
    (OUT/'audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('AUDIT PASS: 4 finished runs, best hashes and all sealed sources unchanged',flush=True)
    return record
if __name__=='__main__':
    record=audit()
    if '--audit-only' not in sys.argv:
        for filename in ('suite_failure.txt',):
            p=BASE/filename
            if p.exists():p.rename(OUT/(filename+'.original'))
        try:runpy.run_path(str(PERF/'70_c2ms_dilated_backbone/provenance/run_suite.py'),run_name='__main__')
        finally:
            after={p:{'best':sha(Path(p)/'checkpoints/best.pt'),'history':sha(Path(p)/'history.json')} for p in record['before']}
            assert all(v['best']==record['before'][p]['best_sha256'] and v['history']==record['before'][p]['history_sha256'] for p,v in after.items())
            (OUT/'unchanged_training.json').write_text(json.dumps(after,indent=2))
