"""One finite reassessment run: transfer private files and restart bounded CI batches.

Launch this script in a detached process. It does not create a service or a
scheduled automation. Registration tokens stay in memory and are never logged.
"""
import argparse,base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO='BIYONGHIYON/CtrS'

def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+'.part');t.write_text(json.dumps(d,indent=2),encoding='utf8');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def command(args,**kw):return subprocess.check_output(args,text=True,**kw)
def remote(a,code):
 enc=base64.b64encode(code.encode()).decode()
 return command(['ssh','-S',a.socket,a.host,f'C:/CtrS/.venv/Scripts/python.exe -c "import base64;exec(base64.b64decode(\'{enc}\'))"'])
def snapshot(a):
 code="from pathlib import Path\nimport json\nr=Path('C:/CtrS-matlab-reassessment/v2')\nprint(json.dumps(dict(records=[json.loads(p.read_text(encoding='utf8')) for p in (r/'records').glob('*.json')],status=json.loads((r/'status.json').read_text(encoding='utf8')) if (r/'status.json').exists() else {})))"
 return json.loads(remote(a,code))
def transfer(a,name,expected):
 p=a.work/name
 if p.exists() and sha(p)==expected:return
 p.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(p)+'.part')
 subprocess.run(['scp','-o','ControlPath='+a.socket,a.host+':C:/CtrS-matlab-reassessment/v2/'+name,str(temp)],check=True,stdout=subprocess.DEVNULL)
 if sha(temp)!=expected:raise ValueError('Transfer checksum mismatch: '+name)
 temp.replace(p)
def pending(a):
 n=0
 for p in (a.work/'records').glob('*.json'):
  rec=json.loads(p.read_text(encoding='utf8'))
  for e in rec.get('exports',[]):
   source=a.work/'queue'/e['file'];target=a.work/'done'/(e['file']+'.json')
   if not source.exists():continue
   if not target.exists():n+=1;continue
   d=json.loads(target.read_text())
   if d.get('contract_id')!=e['contract_id'] or len(d.get('rows',[]))!=e['scenes']:n+=1
 return n

def batch(a):
 # Reuse the one user-approved temporary runner installation, no service.
 # Runner configuration is removed by --ephemeral after each completed job.
 runner=a.runner
 if (runner/'.runner').exists():
  token=json.loads(command(['gh','api','--method','POST',f'repos/{REPO}/actions/runners/remove-token']))['token']
  subprocess.run(['./config.sh','remove','--token',token],cwd=runner,check=True,stdout=subprocess.DEVNULL)
 token=json.loads(command(['gh','api','--method','POST',f'repos/{REPO}/actions/runners/registration-token']))['token']
 name='ctrs-matlab-once-'+str(int(time.time()))
 subprocess.run(['./config.sh','--unattended','--ephemeral','--disableupdate','--url','https://github.com/'+REPO,'--token',token,'--name',name,'--labels','ctrs-matlab-reassessment','--work','_work'],cwd=runner,check=True,stdout=subprocess.DEVNULL)
 log=(a.work/'runner.log').open('ab')
 listener=subprocess.Popen(['./run.sh'],cwd=runner,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 before=json.loads(command(['gh','run','list','--repo',REPO,'--workflow','matlab-full-reassessment.yml','--branch','metric-reference-audit','--limit','1','--json','databaseId']))
 previous=before[0]['databaseId'] if before else None
 subprocess.run(['gh','workflow','run','matlab-full-reassessment.yml','--repo',REPO,'--ref','metric-reference-audit'],check=True,stdout=subprocess.DEVNULL)
 run=None;start=time.time()
 while time.time()-start<180:
  runs=json.loads(command(['gh','run','list','--repo',REPO,'--workflow','matlab-full-reassessment.yml','--branch','metric-reference-audit','--limit','1','--json','databaseId,status']))
  if runs and runs[0]['databaseId']!=previous:run=runs[0]['databaseId'];break
  time.sleep(10)
 if run is None:raise RuntimeError('Workflow dispatch not confirmed')
 atomic(a.work/'ci.json',dict(run=run,runner_pid=listener.pid,state='running'))
 return listener,log,run

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--host',default='ctrs-server');p.add_argument('--socket',default='/private/tmp/ctrs-frequency-ssh');p.add_argument('--hours',type=float,default=18);a=p.parse_args()
 a.work.mkdir(parents=True,exist_ok=True);(a.work/'queue').mkdir(exist_ok=True)
 start=time.time();active=None;failures=0
 while time.time()-start<a.hours*3600:
  try:
   info=snapshot(a);atomic(a.work/'inference_status.json',info['status'])
   for rec in info['records']:
    key=hashlib.sha256(rec['checkpoint']['path'].encode()).hexdigest()[:20];atomic(a.work/'records'/(key+'.json'),rec)
    if rec['status']!='inference_exported_matlab_pending':continue
    for e in rec['exports']:
     transfer(a,'inputs/'+e['input'],e['input_sha256']);transfer(a,'queue/'+e['file'],e['sha256'])
   if active and active[0].poll() is not None:
    state=json.loads(command(['gh','run','view',str(active[2]),'--repo',REPO,'--json','status,conclusion']))
    atomic(a.work/'ci.json',dict(run=active[2],**state));active[1].close();active=None
    failures+=state.get('conclusion')!='success'
   left=pending(a);finished=info['status'].get('state')=='scan_finished'
   if left and active is None:
    if failures>=3:raise RuntimeError('Three CI batch failures; manual correction required')
    active=batch(a)
   atomic(a.work/'pipeline_status.json',dict(state='running',inference=info['status'],pending_matlab=left,matlab_completed=len(list((a.work/'done').glob('*.json'))),ci_failures=failures))
   if finished and left==0:
    (a.work/'STOP').touch();atomic(a.work/'pipeline_status.json',dict(state='finished',inference=info['status'],complete=info['status'].get('errors')==0,matlab_completed=len(list((a.work/'done').glob('*.json')))));return
  except Exception as e:
   atomic(a.work/'controller_error.json',dict(time=time.time(),error=repr(e)));print(type(e).__name__,str(e)[:200],flush=True)
   if failures>=3:return
  time.sleep(15)
 atomic(a.work/'controller_error.json',dict(error='Finite run deadline reached; exports retained for resume'))
if __name__=='__main__':main()
