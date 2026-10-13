import subprocess,sys,json,time
from pathlib import Path
base=Path(__file__).resolve().parents[3];run=Path(__file__).with_name('run.py');runtime=base/'runtime/seed46';runtime.mkdir(parents=True,exist_ok=True)
def call(mode,*extra):subprocess.run([sys.executable,str(run),mode,'--runtime',str(runtime),*extra],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('verify.py'))],check=True)
call('prepare','--data','C:/CtrS/SSA-MRN/data/dataset')
for micro in [64,32,16]:
 jobs=[]
 for variant in ['original','F0']:
  log=(runtime/f'probe_{variant}_{micro}.log').open('w');p=subprocess.Popen([sys.executable,str(run),'probe','--runtime',str(runtime),'--micro',str(micro),'--variant',variant],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
 exits=[]
 for p,f in jobs:exits.append(p.wait());f.close()
 if all(x==0 for x in exits):
  probes=[json.loads((runtime/f'probe_{v}.json').read_text()) for v in ['original','F0']]
  if sum(p['peak_reserved_mib'] for p in probes)<6400:break
else:raise RuntimeError('No safe two-job common micro-batch')
for sensor in ['QB','GF2','WV3']:
 p=runtime/sensor/'plan.json';obj=json.loads(p.read_text());obj['micro']=micro;p.write_text(json.dumps(obj,indent=2))
p=runtime/'suite_plan.json';obj=json.loads(p.read_text());obj['micro']=micro;obj['probe_results']=probes;p.write_text(json.dumps(obj,indent=2))
print('COMMON MICRO',micro,'EFFECTIVE BATCH64',flush=True)
call('start')
