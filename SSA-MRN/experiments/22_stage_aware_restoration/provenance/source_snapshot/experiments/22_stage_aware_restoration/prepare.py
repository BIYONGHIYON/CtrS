"""Prepare a separate QB cache, profile two simultaneous jobs, verify AMP and launch detached."""
import argparse,hashlib,json,shutil,subprocess,sys,time,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'experiments/22_stage_aware_restoration/run.py'
VARIANTS=['F0','F1','F2','F3','F4','F5']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def command(*args):return [sys.executable,str(SCRIPT),*map(str,args)]
def run_logged(runtime,name,args):
 with (runtime/name).open('w',encoding='utf-8') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
def guard_idle():
 # This prepares a new checkout; never terminate another job to obtain resources.
 text=subprocess.check_output(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\\.exe$'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],text=True).strip()
 if text:
  rows=json.loads(text);rows=[rows] if isinstance(rows,dict) else rows
  others=[x for x in rows if x['ProcessId']!=__import__('os').getpid() and any(w in (x.get('CommandLine') or '').lower() for w in (' train ',' controller ','common/train.py','training.py'))]
  if others:raise RuntimeError('Another training/controller is active: '+json.dumps(others))
def probe_pair(runtime,pair,batch,tag):
 active=[]
 for j,v in enumerate(pair):
  path=runtime/f'probe_{tag}_{j}.log';f=path.open('w',encoding='utf-8')
  p=subprocess.Popen(command('probe','--runtime',runtime,'--variant',v,'--micro',batch,'--real-data','--workers',1),stdout=f,stderr=subprocess.STDOUT);active.append((v,p,f))
 peak=0;util=0
 while any(p.poll() is None for _,p,_ in active):
  q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  try:
   values=q.stdout.strip().splitlines()[0].split(',');peak=max(peak,int(values[0]));util=max(util,int(values[1]))
  except (ValueError,IndexError):pass
  time.sleep(.5)
 processes={}
 for v,p,f in active:
  f.close();path=runtime/f'probe_{v}_{batch}.json'
  processes[v]=json.loads(path.read_text()) if p.returncode==0 and path.exists() else dict(exitcode=p.returncode)
 return dict(batch=batch,pair=pair,gpu_peak_used_mib=peak,gpu_peak_utilization_percent=util,processes=processes,passed=all('exitcode' not in x for x in processes.values()) and peak<=7200)
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--cache-from',type=Path,required=True);p.add_argument('--base-commit',required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
 r=a.runtime
 if (r/'plan.json').exists():raise RuntimeError('Use a fresh runtime; do not overwrite a run')
 r.mkdir(parents=True,exist_ok=True);guard_idle()
 verification=subprocess.check_output([sys.executable,str(ROOT/'experiments/22_stage_aware_restoration/verify.py')],text=True)
 write(r/'model_verification.json',json.loads(verification));print('All six model contracts passed',flush=True)
 # Reuse immutable pre-normalized data using NTFS hardlinks. Every byte is rechecked.
 source=a.cache_from;manifest=json.loads((source/'data_manifest.json').read_text())
 expected={'train':'7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371','val':'b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470'}
 for split,rec in manifest.items():
  if rec['sha256']!=expected[split]:raise RuntimeError('Changed source data')
  if sha(Path(rec['path']))!=rec['sha256']:raise RuntimeError('H5 bytes changed')
  d=r/'cache'/split;d.mkdir(parents=True)
  for k,digest in rec['cache_hashes'].items():
   src=source/'cache'/split/(k+'.npy');dst=d/(k+'.npy')
   if sha(src)!=digest:raise RuntimeError('Changed cache '+str(src))
   __import__('os').link(src,dst)
 write(r/'data_manifest.json',manifest);print('Full QB cache hashes passed; hardlinks ready',flush=True)
 names=['experiments/22_stage_aware_restoration/'+n for n in ('run.py','model.py','prepare.py','verify.py')]+['experiments/20_recurrent_guidance/model.py','common/model.py','common/metrics.py','common/preview.py','references/upstream/network.py']
 plan=dict(variants=VARIANTS,epochs=50,effective_batch=96,micro=96,workers=1,K=4,seed=45,lr=1e-4,loss='MSE',amp=True,max_parallel=2,data=str(a.data),cache_root=str(r/'cache'),source_base_commit=a.base_commit,source_hashes={n:sha(ROOT/n) for n in names},selection='validation MSE; SAM only exact tie; earlier best epoch',data_hashes=expected,architecture='F0 E1 control; F1 stage alpha; F2 zero-init stage affine; F3 both; F4 F3 plus final-only high-pass PAN; F5 single restoration',observation='no new observation correction; bilinear FR diagnostic only',queue_order=VARIANTS)
 write(r/'plan.json',plan)
 records=[]
 # Fixed effective batch96; reduce micro only if safety requires it.
 for micro in (96,48):
  pair=probe_pair(r,['F3','F4'],micro,'heavy_'+str(micro));records.append(pair)
  print('Heavy pair micro',micro,'GPU MiB',pair['gpu_peak_used_mib'],'passed',pair['passed'],flush=True)
  if pair['passed']:break
 else:raise RuntimeError('Cannot fit two jobs safely; training withheld')
 plan['micro']=micro;write(r/'plan.json',plan)
 for pair in (['F0','F1'],['F2','F3'],['F4','F5']):
  rec=probe_pair(r,pair,micro,'confirm_'+'_'.join(pair));records.append(rec)
  if not rec['passed']:raise RuntimeError('Pair does not fit safely; training withheld')
 write(r/'benchmark.json',dict(records=records,effective_batch=96,micro=micro,workers_per_process=1,memory_limit_mib=7200))
 smoke=r.parent/'stage_aware_smoke'
 if smoke.exists():raise RuntimeError('Smoke runtime exists')
 smoke.mkdir();write(smoke/'plan.json',dict(plan,epochs=1,effective_batch=4,micro=4,workers=0,cache_root=str(smoke/'cache')))
 import numpy as np,torch,importlib
 for split,n in [('train',8),('val',4)]:
  d=smoke/'cache'/split;d.mkdir(parents=True)
  for k in ('pan','lms','ms','gt'):np.save(d/(k+'.npy'),np.asarray(np.load(r/'cache'/split/(k+'.npy'),mmap_mode='r')[:n]))
 results={};sys.path.insert(0,str(ROOT));model_cls=importlib.import_module('experiments.22_stage_aware_restoration.model').StageAwareNet
 for v in VARIANTS:
  run_logged(smoke,v+'.log',command('train','--runtime',smoke,'--variant',v))
  ck=torch.load(smoke/v/'latest.pt',map_location='cpu',weights_only=False)
  assert ck['epoch']==1 and ck['skipped_steps']==0
  m=model_cls(v);m.load_state_dict(ck['model'],strict=True)
  results[v]=dict(epoch=1,val_mse=ck['best_mse'],strict_load=True,skipped_steps=0)
 write(r/'training_smoke.json',results)
 # Exercise full RR/FR evaluator on smoke weights BEFORE starting the actual suite.
 run_logged(smoke,'evaluate.log',command('evaluate','--runtime',smoke))
 report=json.loads((smoke/'evaluation/metrics.json').read_text());assert len(report['runs'])==6
 assert len(list((smoke/'evaluation').glob('F*_scene_*.png')))==30
 write(r/'evaluation_preflight.json',dict(runs=6,rr_fr_scene_rows=240,images=30,full_evaluation_passed=True,results_are_smoke_not_research=True))
 for n in names:
  dst=r/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
 print('Real-data AMP smoke and complete RR/FR evaluator passed',flush=True)
 if a.launch:
  guard_idle();subprocess.run(command('start','--runtime',r),check=True);time.sleep(20)
  q=json.loads((r/'queue.json').read_text());assert set(q['active'])=={'F0','F1'} and not q['failed']
  for v in ('F0','F1'):
   s=json.loads((r/v/'status.json').read_text());assert s['state'] in ('training','validation');print(v,json.dumps(s),flush=True)
  write(r/'launch_verification.json',dict(time=time.time(),active=['F0','F1'],detached_controller=True,source_hashes=plan['source_hashes']))
if __name__=='__main__':main()
