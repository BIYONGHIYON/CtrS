"""Two independent sensors on two GPUs. One phase per invocation, bounded by six hours."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,html,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from utils import digest,write
HERE=Path(__file__).resolve().parent
NAME='U2Net_QBGF2_retrain_v1'

def restore(reference,dest,expected):
    reference=Path(reference)
    if reference.is_file():
        with zipfile.ZipFile(reference) as z:
            records=[n for n in z.namelist() if n.endswith('experiment.json')]
            if len(records)!=1:raise ValueError('Expected exactly one experiment.json in resume ZIP')
            prefix=records[0][:-len('experiment.json')];meta=json.loads(z.read(records[0]))
            if meta['package_sha256']!=expected:raise ValueError('Resume ZIP belongs to different code/config')
            if dest.exists():raise FileExistsError('Resume destination exists; use working root explicitly')
            dest.mkdir(parents=True)
            for n in z.namelist():
                if not n.startswith(prefix) or n.endswith('/'):continue
                rel=n[len(prefix):];p=(dest/rel).resolve()
                if not p.is_relative_to(dest.resolve()):raise ValueError('Unsafe ZIP path')
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(n) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
    else:
        meta=json.loads((reference/'experiment.json').read_text(encoding='utf8'))
        if meta['package_sha256']!=expected:raise ValueError('Resume folder belongs to different code/config')
        if reference.resolve()!=dest.resolve():
            if dest.exists():raise FileExistsError('Destination exists; nothing overwritten')
            shutil.copytree(reference,dest)
    archive_manifest=dest/'artifact_manifest.json'
    if archive_manifest.exists():
        for rel,rec in json.loads(archive_manifest.read_text(encoding='utf8'))['files'].items():
            if digest(dest/rel)!=rec['sha256']:raise ValueError('Restored artifact hash mismatch: '+rel)
    return dest

def resolve_resume(reference,input_root):
    if reference!='AUTO':return Path(reference)
    roots=[p.parent for p in Path(input_root).rglob('experiment.json') if p.parent.name==NAME]
    if len(roots)==1:return roots[0]
    if len(roots)>1:raise ValueError('Multiple prior outputs attached; set RESUME_FROM to the intended root')
    zs=list(Path(input_root).rglob(NAME+'_results.zip'))
    if len(zs)!=1:raise ValueError('Attach the latest cumulative results ZIP/folder, or set RESUME_FROM explicitly')
    return zs[0]

def graphs(root):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.plotcache'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for phase in ('smoke','train'):
        fig,grid=plt.subplots(2,3,figsize=(13,7));ax=grid.ravel();any_rows=False
        for sensor in ('QB','GF2'):
            p=root/phase/sensor/'history.json'
            if p.exists():
                r=json.loads(p.read_text(encoding='utf8'))
                if r:
                    any_rows=True;epochs=[x['epoch'] for x in r]
                    values=[('train L1',[x['train_l1'] for x in r]),('train MSE',[x['train_mse'] for x in r]),('val MSE',[x['validation']['mse'] for x in r]),('val PSNR sensor peak',[x['validation']['psnr_sensor_peak'] for x in r]),('val SAM degrees',[x['validation']['sam_degrees'] for x in r]),('learning rate',[x['lr'] for x in r])]
                    for axis,(title,ys) in zip(ax,values):axis.plot(epochs,ys,label=sensor);axis.set_title(title)
        if any_rows:
            for a in ax:a.legend();a.set_xlabel('Epoch');a.grid(alpha=.2)
            fig.tight_layout();fig.savefig(root/(phase+'_curves.png'),dpi=140)
        plt.close(fig)

def package(root):
    graphs(root)
    files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in root.rglob('*') if p.is_file() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(root).parts) and p.name!='artifact_manifest.json'}
    write(root/'artifact_manifest.json',dict(files=files))
    archive=root.with_name(NAME+'_results.zip');tmp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*files,'artifact_manifest.json']:z.write(root/rel,Path(NAME)/rel)
    tmp.replace(archive);print('CUMULATIVE ZIP:',archive,'SHA256:',digest(archive),flush=True)

def dashboard(jobs,handle=None):
    def fmt(x):return f'{x:.5g}' if isinstance(x,(int,float)) else x
    rows=[]
    for j in jobs:
        p=Path(j['folder'])/'status.json'
        try:s=json.loads(p.read_text(encoding='utf8'))
        except (OSError,ValueError):s={}
        rows.append('<tr>'+''.join('<td style="padding:8px">'+html.escape(str(fmt(v)))+'</td>' for v in [j['sensor'],'GPU '+str(j['physical_gpu']),s.get('phase','starting'),f"{s.get('epoch',0)}/{s.get('epochs','—')}",f"{s.get('batch',0)}/{s.get('batches','—')}",s.get('train_l1','—'),s.get('val_mse',s.get('best_val_mse','—')),s.get('gpu_memory_gib','—'),s.get('remaining_hours','—')])+'</tr>')
    content='<div style="border:1px solid #ccd7e5;padding:14px;border-radius:10px"><b>U2Net · QB / GF2 · seed42</b><p>Independent training · FP32 · effective16 / micro4 · session ≤6h · ETA excludes test/images</p><table><tr>'+''.join('<th style="padding:8px">'+x+'</th>' for x in ['Sensor','Device','Status','Epoch','Batch','Train L1','Val MSE','Peak GiB','ETA hours'])+'</tr>'+''.join(rows)+'</table></div>'
    try:
        from IPython.display import display,HTML
        if handle is None:return display(HTML(content),display_id=True)
        handle.update(HTML(content))
    except Exception:pass
    return handle

def launch(jobs,stage,mode,max_updates=None):
    stop=stage/'stop.request';stop.unlink(missing_ok=True);processes=[];logs=[];handle=None
    try:
        for j in jobs:
            folder=Path(j['folder']);folder.mkdir(parents=True,exist_ok=True)
            if (folder/'error.json').exists():raise RuntimeError('Previous worker error exists: inspect '+str(folder/'error.json')+' before retry; use a new run for changed settings')
            j['stop_file']=str(stop);jobfile=folder/'job.json';write(jobfile,j)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(j['physical_gpu']);env['OMP_NUM_THREADS']='2';env['PYTHONDONTWRITEBYTECODE']='1'
            log=(folder/(mode+'.log')).open('a',encoding='utf8');logs.append(log)
            cmd=[sys.executable,str(HERE/('train.py' if mode=='train' else 'evaluate.py')),'--job',str(jobfile)]
            if mode=='train' and max_updates is not None:cmd+=['--max-updates',str(max_updates)]
            if mode=='evaluate' and j['config'].get('smoke_mode'):cmd+=['--smoke']
            processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True))
        while any(p.poll() is None for p in processes):handle=dashboard(jobs,handle);time.sleep(5)
        dashboard(jobs,handle)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Worker failed. See sensor error.json and log; existing checkpoints preserved')
    finally:
        stop.touch()
        # Graceful pause requested at optimizer update boundary, then bounded wait.
        until=time.time()+45
        while any(p.poll() is None for p in processes) and time.time()<until:time.sleep(1)
        for p in processes:
            if p.poll() is None:p.terminate()
        for log in logs:log.close()

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['smoke','train','test'],required=True);a.add_argument('--action',choices=['new','resume'],default='resume')
    a.add_argument('--resume-from',default='AUTO');a.add_argument('--input',type=Path,default=Path('/kaggle/input'));a.add_argument('--output',type=Path,default=Path('/kaggle/working')/NAME);a.add_argument('--hours',type=float,default=6)
    args=a.parse_args();start=time.time()
    if not 0<args.hours<=6:raise ValueError('Session budget must be >0 and ≤6 hours')
    manifest=json.loads((HERE/'package_manifest.json').read_text(encoding='utf8'));source_sha=digest(HERE/'package_manifest.json')
    for rel,r in manifest['files'].items():
        if digest(HERE/rel)!=r['sha256']:raise ValueError('Package changed: '+rel)
    root=args.output
    if args.action=='new':
        if root.exists():raise FileExistsError('Run exists: resume explicitly; nothing overwritten')
        root.mkdir(parents=True);write(root/'experiment.json',dict(name=NAME,package_sha256=source_sha,created=time.time()))
    elif not root.exists():restore(resolve_resume(args.resume_from,args.input),root,source_sha)
    meta=json.loads((root/'experiment.json').read_text(encoding='utf8'))
    if meta['package_sha256']!=source_sha:raise ValueError('Run source/config differs; do not resume old weights with new code')
    retained=root/'source'
    if not retained.exists():shutil.copytree(HERE,retained,ignore=shutil.ignore_patterns('__pycache__'))
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Select Kaggle GPU T4×2')
    devices=[dict(index=i,name=torch.cuda.get_device_name(i),total_gib=torch.cuda.get_device_properties(i).total_memory/1024**3) for i in range(2)]
    if devices[0]['name']!=devices[1]['name'] or devices[0]['total_gib']!=devices[1]['total_gib']:raise ValueError('GPU environments differ')
    import platform,numpy,h5py
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),numpy=numpy.__version__,h5py=h5py.__version__,devices=devices,precision='FP32',amp=False,tf32=False)
    if (root/'environment.json').exists() and json.loads((root/'environment.json').read_text(encoding='utf8'))!=environment:
        raise ValueError('GPU/torch/CUDA environment differs from the saved run; inspect before resuming')
    write(root/'environment.json',environment)
    if args.phase=='train' and not all((root/'smoke'/s/'smoke_complete.json').exists() for s in ('QB','GF2')):raise ValueError('Finish smoke on both sensors first; attach its cumulative ZIP in a new session')
    stage=root/('train' if args.phase=='test' else args.phase);stage.mkdir(parents=True,exist_ok=True)
    jobs=[]
    try:
        for i,sensor in enumerate(('QB','GF2')):
            c=dict(cfg)
            if args.phase=='smoke':c['smoke_mode']=True;c['epochs']=cfg['smoke']['epochs'];c['checkpoint_every_updates']=1
            folder=stage/sensor
            if args.phase!='test':
                from data import prepare
                data=prepare(args.input,root/'.cache',sensor)
                compact={k:{kk:vv for kk,vv in v.items() if kk!='source'} for k,v in data.items()}
                proof=dict(data=compact,source_sha256=source_sha);write(folder/'data_manifest.json',proof)
            else:
                if not (folder/'training_complete.json').exists():raise ValueError('Finish both sensor training before final test')
                proof=json.loads((folder/'data_manifest.json').read_text(encoding='utf8'))
            jobs.append(dict(sensor=sensor,folder=str(folder.resolve()),cache=str((root/'.cache').resolve()),input=str(args.input.resolve()),physical_gpu=i,device='cuda:0',config=c,proof=proof,deadline=start+args.hours*3600-min(900,args.hours*3600*.1)))
        if args.phase=='smoke':
            if not any((Path(j['folder'])/'latest.pt').exists() for j in jobs):launch(jobs,stage,'train',max_updates=1)
            launch(jobs,stage,'train')
            launch(jobs,stage,'evaluate')
            for j in jobs:
                folder=Path(j['folder']);ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
                if ck['state']['resumes']<1:raise ValueError('Smoke did not test resume')
                test_path=folder/'evaluation/metrics_smoke.json'
                if not test_path.exists():raise ValueError('Smoke evaluation did not complete; resume this phase')
                test=json.loads(test_path.read_text(encoding='utf8'))
                if len(test['rr'])!=1 or len(test['fr'])!=1 or not test['smoke_excluded']:raise ValueError('Smoke evaluation incomplete')
                write(folder/'smoke_complete.json',dict(resume_checked=True,training_and_evaluation_checked=True,excluded_from_final=True))
        elif args.phase=='train':launch(jobs,stage,'train')
        else:launch(jobs,stage,'evaluate')
    finally:package(root)

if __name__=='__main__':main()
