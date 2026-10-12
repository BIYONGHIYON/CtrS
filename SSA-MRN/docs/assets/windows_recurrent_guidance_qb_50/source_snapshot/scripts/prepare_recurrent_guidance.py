"""Prepare a separate QB cache, profile two simultaneous jobs, verify AMP and launch detached."""
import argparse,hashlib,json,shutil,subprocess,sys,time,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/windows_recurrent_guidance_suite.py'
VARIANTS=['D0','D1','D2','D3','D4','D5','D6']
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
  others=[x for x in rows if x['ProcessId']!=__import__('os').getpid() and any(w in (x.get('CommandLine') or '').lower() for w in (' train ',' controller ','train.py','training.py'))]
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
 parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--base-commit',required=True);parser.add_argument('--launch',action='store_true');a=parser.parse_args();r=a.runtime
 if r.exists() and any((r/v/'latest.pt').exists() for v in VARIANTS):raise RuntimeError('Use resume-latest for existing training, not preparation')
 r.mkdir(parents=True,exist_ok=True);guard_idle()
 verification=subprocess.check_output([sys.executable,str(ROOT/'scripts/verify_recurrent_guidance.py')],text=True);write(r/'model_verification.json',json.loads(verification))
 print('Model equivalence, live paths and recovery verified',flush=True)
 run_logged(r,'cache.log',command('cache','--runtime',r,'--data',a.data))
 print('Full QB mmap cache complete',flush=True)
 manifest=json.loads((r/'data_manifest.json').read_text())
 expected={'train':'7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371','val':'b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470'}
 for split,rec in manifest.items():
  assert rec['sha256']==expected[split]
  for k,digest in rec['cache_hashes'].items():assert sha(r/'cache'/split/(k+'.npy'))==digest
 records=[];valid=[]
 for batch in (32,64,96):
  rec=probe_pair(r,['D5','D6'],batch,f'heavy_{batch}');records.append(rec)
  print('Heavy pair',batch,rec['gpu_peak_used_mib'],'MiB, passed',rec['passed'],flush=True)
  if not rec['passed']:break
  rates=[x['patches_per_second'] for x in rec['processes'].values()]
  valid.append(dict(batch=batch,score=1/sum(1/x for x in rates),peak_mib=rec['gpu_peak_used_mib']))
 if not valid:
  rec=probe_pair(r,['D5','D6'],16,'heavy_16');records.append(rec)
  if not rec['passed']:raise RuntimeError('Even batch16 failed safety profile; no training launched')
  rates=[x['patches_per_second'] for x in rec['processes'].values()];valid.append(dict(batch=16,score=1/sum(1/x for x in rates),peak_mib=rec['gpu_peak_used_mib']))
 chosen=max(valid,key=lambda x:x['score']);batch=chosen['batch']
 # Every arm gets the same actual-data AMP+FP32 probe in a concurrent pair.
 for pair in [['D0','D1'],['D2','D3'],['D4','D5'],['D5','D6']]:
  rec=probe_pair(r,pair,batch,'confirm_'+'_'.join(pair));records.append(rec)
  if not rec['passed']:raise RuntimeError('Common batch failed full confirmation; no training launched')
 write(r/'benchmark.json',dict(records=records,valid=valid,chosen=chosen,workers_per_process=1,rule='fastest safe common batch; actual data, AMP train and FP32 validation, total GPU <=7200MiB'))
 names=['scripts/windows_recurrent_guidance_suite.py','scripts/prepare_recurrent_guidance.py','scripts/verify_recurrent_guidance.py','src/ssamrn/models/recurrent_guidance.py','src/ssamrn/models/ssa_mrn.py','references/upstream/network.py','src/ssamrn/metrics.py','scripts/evaluate_a6000.py']
 plan=dict(variants=VARIANTS,epochs=50,effective_batch=batch,micro=batch,workers=1,K=4,seed=43,lr=1e-4,loss='MSE',amp=True,max_parallel=2,data=str(a.data),source_base_commit=a.base_commit,source_hashes={n:sha(ROOT/n) for n in names},selection='validation MSE; SAM only exact tie; earlier best epoch',data_hashes=expected,architecture='per-band SSAI retained; shared stages / differentiable memory / observation error / PAN feature difference',observation='fixed bilinear scale4 align_corners=True, not sensor-calibrated MTF',queue_order=VARIANTS)
 write(r/'plan.json',plan)
 smoke=r.parent/'recurrent_guidance_smoke'
 if smoke.exists():raise RuntimeError('Smoke folder already exists; use a fresh runtime')
 smoke.mkdir();write(smoke/'plan.json',dict(plan,epochs=1,effective_batch=4,micro=4,workers=0))
 import numpy as np,torch
 for split,n in [('train',8),('val',4)]:
  d=smoke/'cache'/split;d.mkdir(parents=True)
  for k in ('pan','lms','ms','gt'):np.save(d/(k+'.npy'),np.asarray(np.load(r/'cache'/split/(k+'.npy'),mmap_mode='r')[:n]))
 results={}
 sys.path.insert(0,str(ROOT/'src'));from ssamrn.models.recurrent_guidance import GuidanceNet
 for v in VARIANTS:
  run_logged(smoke,v+'.log',command('train','--runtime',smoke,'--variant',v));ck=torch.load(smoke/v/'latest.pt',map_location='cpu',weights_only=False)
  assert ck['epoch']==1 and ck['next_batch']==0 and len(ck['history'])==1 and ck['skipped_steps']==0 and ck['best_epoch']==1
  model=GuidanceNet(v);model.load_state_dict(ck['model'],strict=True)
  results[v]=dict(epoch=1,val_mse=ck['best_mse'],skipped_steps=ck['skipped_steps'],best_sha256=sha(smoke/v/'best.pt'),latest_sha256=sha(smoke/v/'latest.pt'),strict_load=True)
 write(r/'training_smoke.json',results)
 for n in names:
  p=r/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
 write(r/'cleanup.json',dict(previous_runtime='C:/CtrS-dual-stream-suite/runtime/qb_dual_stream_50',previous_checkout_exists=Path('C:/CtrS-dual-stream-suite').exists(),deleted_paths=[],recovered_bytes=0,reason='Previous separate experiment checkout already absent before this task; original CtrS/data/venv retained',recovery_branch='ssa-dual-stream-suite',recovery_commit='5588699009249c4377ff64cc819ff18090d3ab30',recovery_pr=54))
 print('Common batch',batch,'all seven real AMP smoke runs passed',flush=True)
 if a.launch:
  guard_idle();subprocess.run(command('start','--runtime',r),check=True);time.sleep(15)
  q=json.loads((r/'queue.json').read_text());assert set(q['active'])=={'D0','D1'} and not q['failed']
  for v in ('D0','D1'):
   status=json.loads((r/v/'status.json').read_text());assert status['state'] in ('training','validation');print(v,json.dumps(status),flush=True)
  write(r/'launch_verification.json',dict(time=time.time(),active=['D0','D1'],detached_controller=True,source_hashes=plan['source_hashes']))
if __name__=='__main__':main()
