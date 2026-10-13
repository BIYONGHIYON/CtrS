import os,sys,json,time,hashlib,shutil,subprocess,zipfile,signal,math,csv
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert PHASE in ('smoke','probe','select','final_train','final_test')
assert SEED in (42,43,44) and 0<MAX_SESSION_HOURS<=6
assert RUN_TAG.startswith('EMA_1st_') and all(c.isalnum() or c in '_-' for c in RUN_TAG)
output=Path('/kaggle/working')/RUN_TAG
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def verify(root):
    e=read(root/'experiment.json')
    if e['name']!='EMA_1st' or e['tag']!=RUN_TAG:raise ValueError('Different experiment; nothing copied')
    m=read(root/'artifact_manifest.json')
    for name,v in m.items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=v['sha256']:raise ValueError('Archive evidence mismatch: '+name)
    return e
if not output.exists() and PHASE!='smoke':
    if RESUME_FROM:
        origin=Path(RESUME_FROM);verify(origin);shutil.copytree(origin,output)
    else:
        attached=list(Path('/kaggle/input').rglob(SOURCE_ZIP)) if SOURCE_ZIP else []
        if len(attached)==1:
            temp=Path('/kaggle/working')/(RUN_TAG+'_restore_'+str(time.time_ns()));temp.mkdir()
            with zipfile.ZipFile(attached[0]) as z:
                if z.testzip():raise ValueError('ZIP CRC error')
                for n in z.namelist():
                    if not (temp/n).resolve().is_relative_to(temp.resolve()):raise ValueError('Unsafe ZIP path')
                z.extractall(temp)
            roots=list(temp.rglob('experiment.json'))
            if len(roots)!=1:raise ValueError('Expected one experiment root')
            verify(roots[0].parent);shutil.copytree(roots[0].parent,output)
        else:
            roots=[p.parent for p in Path('/kaggle/input').rglob('experiment.json')
                   if read(p).get('name')=='EMA_1st' and read(p).get('tag')==RUN_TAG]
            if len(roots)!=1:raise ValueError('Set RESUME_FROM to exact attached folder containing experiment.json. Attach ONLY latest cumulative output.')
            verify(roots[0]);shutil.copytree(roots[0],output)
if output.exists() and not (output/'experiment.json').exists():
    raise FileExistsError('Existing directory is not a valid EMA experiment. Use a new RUN_TAG; nothing overwritten.')
output.mkdir(parents=True,exist_ok=True)
missing=[p for m,p in [('numpy','numpy'),('h5py','h5py'),('scipy','scipy'),('skimage','scikit-image'),('matplotlib','matplotlib')] if __import__('importlib.util',fromlist=['find_spec']).find_spec(m) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
import numpy as np,h5py,torch
code=output/'source';code.mkdir(exist_ok=True)
for name,value in SOURCES.items():
    p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_text(encoding='utf-8')!=value:raise ValueError('Frozen source differs: '+name)
    p.write_text(value,encoding='utf-8')
sys.path.insert(0,str(code));sys.path.insert(0,str(code/'SSA-MRN/src'))
from ema_dashboard import Dashboard
from ema_worker import state_hash
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
dashboard=Dashboard(PHASE,output)
TRAIN_SEED=SEED;SEED=42
DATA_ROOT='/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset'
MODE='smoke' if PHASE=='smoke' else 'full'
exec(SOURCES['ema_preflight.py'])
SEED=TRAIN_SEED
write(output/('data_manifest_'+MODE+'.json'),scientific_data)
env={'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'h5py':h5py.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
     'gpu':[{'index':g,'name':torch.cuda.get_device_name(g),'total_gib':torch.cuda.get_device_properties(g).total_memory/(1<<30),
             'capability':list(torch.cuda.get_device_capability(g))} for g in range(torch.cuda.device_count())]}
if len(env['gpu'])!=2 or env['gpu'][0]['name']!=env['gpu'][1]['name'] or env['gpu'][0]['total_gib']!=env['gpu'][1]['total_gib']:
    raise RuntimeError('Require two matching GPUs; no silent fallback')
write(output/'environment_current.json',env)
try:(output/('nvidia_smi_'+str(time.time_ns())+'.txt')).write_text(subprocess.check_output(['nvidia-smi'],text=True),encoding='utf-8')
except (OSError,subprocess.SubprocessError):pass
raw={s:{'sha256':v['sha256'],'arrays':{k:{'shape':a['shape'],'dtype':a['dtype']} for k,a in v['arrays'].items()}} for s,v in scientific_data.items()}
protocol={'name':'EMA_1st','tag':RUN_TAG,'baseline_audit':BASELINE_AUDIT,'sources':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},
    'data':raw,'k':4,'micro':32,'effective_batch':32,'epochs':100,'lr':1e-4,'optimizer':'Adam defaults','precision':'FP32; AMP/TF32 off; deterministic',
    'beta_candidates':[0.99,0.999],'beta_probe_epochs':10,'beta_selection':'lowest EMA validation MSE across 10 probe epochs; tie larger beta',
    'ema_initialization':'exact initial online weights','update':'after successful optimizer step from step1; constant beta; no warmup or feedback',
    'checkpoint_selection':'each arm lowest validation MSE of its evaluated weights; ties earlier epoch',
    'scene_indices':selection['indices'],'seed_list':[42,43,44],
    'environment':{k:env[k] for k in ('torch','numpy','h5py','cuda','cudnn','gpu')}}
basis=digest(protocol)
if (output/'experiment.json').exists():
    old=read(output/'experiment.json')
    if old['basis']!=basis:raise ValueError('Science/data/software/GPU conditions differ. Existing evidence preserved; start a new RUN_TAG.')
else:write(output/'experiment.json',{**protocol,'basis':basis})
write(output/'selection_scenes.json',selection)
deadline=time.time()+MAX_SESSION_HOURS*3600
phasekey=PHASE+('_seed_'+str(SEED) if PHASE.startswith('final_') else '')
stage=output/'sessions'/phasekey;stage.mkdir(parents=True,exist_ok=True)
def task(family,seed,label,gpu,betas,selected,epochs,job_type='train',stop=0,reference=False):
    folder=output/family/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    initial=folder/'initial.pt'
    if not initial.exists():
        if job_type=='test' or any(folder.rglob('latest.pt')):raise ValueError('Saved initial state missing')
        torch.manual_seed(seed);m=RestoredPansharpeningNet(channels=4,ssai_dimension=4)
        torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},initial)
        (folder/'initial.sha256').write_text(sha(initial));del m
    if sha(initial)!=(folder/'initial.sha256').read_text():raise ValueError('Initial weights hash differs')
    sealhash=sha(output/'final_settings.json') if family=='final' else None
    cond={'basis':basis,'family':family,'seed':seed,'label':label,'epochs':epochs,'betas':betas,'selected_beta':selected,
          'initial_sha256':sha(initial),'seal_sha256':sealhash,'samples':{s:data[s]['used_samples'] for s in ('train','val')}}
    cfg={'output':str(folder),'code_root':str(code),'conditions':cond,'fingerprint':digest(cond),'data':data,'selection':selection,'seed':seed,'epochs':epochs,
         'batch':32,'micro':32,'amp':False,'lr':1e-4,'deadline':deadline,'reserve_gib':4,'log_seconds':20,'weight':0.,
         'betas':betas,'selected_beta':selected,'job_type':job_type,'stop_after_group':stop}
    path=folder/(label+'_config.json')
    if path.exists() and read(path)['fingerprint']!=cfg['fingerprint']:raise ValueError('Job config mismatch')
    write(path,cfg)
    return {'gpu':gpu,'label':label,'folder':folder/label,'config':cfg,'path':path}
def run(jobs):
    processes=[]
    try:
        for j in jobs:
            j['folder'].mkdir(exist_ok=True);dashboard.starting(j,0)
            dashboard.rows[j['gpu']]['context']=f"{PHASE} · β={j['config']['selected_beta']} · {j['folder'].parent.name}"
            log=open(j['folder']/'worker.log','a',encoding='utf-8');environ=os.environ.copy();environ['CUDA_VISIBLE_DEVICES']=str(j['gpu'])
            p=subprocess.Popen([sys.executable,str(code/'ema_worker.py'),'--config',str(j['path']),'--label',j['label'],'--gpu',str(j['gpu'])],env=environ,stdout=log,stderr=log)
            processes.append((p,j,log,time.monotonic()))
        while any(p.poll() is None for p,_,_,_ in processes):
            for p,j,_,_ in processes:
                if (j['folder']/'status.json').exists():dashboard.observe(j['gpu'],read(j['folder']/'status.json'))
                if p.poll() not in (None,0):raise RuntimeError('Worker failed; see '+str(j['folder']/'worker.log'))
            dashboard.refresh();time.sleep(5)
        for p,j,log,tick in processes:
            if p.returncode:raise RuntimeError('Worker failed; see '+str(j['folder']/'worker.log'))
            write(j['folder']/('session_'+str(time.time_ns())+'.json'),{'wall_seconds':time.monotonic()-tick,'job_type':j['config']['job_type'],'gpu':j['gpu']})
            dashboard.finish(j,p.returncode)
    finally:
        for p,_,log,_ in processes:
            if p.poll() is None:p.send_signal(signal.SIGINT)
        wait_deadline=time.monotonic()+50
        for p,_,log,_ in processes:
            if p.poll() is None:
                try:p.wait(timeout=max(.1,wait_deadline-time.monotonic()))
                except subprocess.TimeoutExpired:p.terminate();p.wait()
            log.close()
        dashboard.refresh()
def pair_check(jobs):
    paths=[j['folder'] for j in jobs]
    if read(paths[0]/'initialization.json')!=read(paths[1]/'initialization.json'):raise ValueError('Initialization differs')
    if read(paths[0]/'orders.json')!=read(paths[1]/'orders.json'):raise ValueError('Batch order differs')
    ck=[torch.load(p/'latest.pt',map_location='cpu',weights_only=False) for p in paths]
    if (ck[0]['epoch'],ck[0]['next_sample_position'])!=(ck[1]['epoch'],ck[1]['next_sample_position']):raise ValueError('Incomplete pair')
    exact=all(torch.equal(v,ck[1]['model'][k]) for k,v in ck[0]['model'].items())
    write(paths[0].parent/'pair_checks.json',{'initial_equal':True,'order_equal':True,'online_weights_exact_equal':exact})
    if not exact:raise ValueError('Online weights mismatch; investigate before interpreting EMA')
def complete(jobs,test=False):return all((j['folder']/('evaluation_complete.json' if test else 'training_complete.json')).exists() for j in jobs)
def curves():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for p in output.rglob('history.json'):
        h=read(p)
        if not h:continue
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        for ax,key in zip(axes,('train_mse','val_mse','val_sam')):ax.plot([r['epoch'] for r in h],[r[key] for r in h]);ax.set_title(key);ax.set_xlabel('epoch')
        fig.tight_layout();fig.savefig(p.parent/'curves.png',dpi=140);plt.close(fig)
def summarize():
    rows=[];deltas=[]
    for seed in (42,43,44):
        root=output/'final'/f'seed_{seed}';files=[root/a/'results.json' for a in ('baseline','ema')]
        if not all(p.exists() for p in files):continue
        rs=[read(p) for p in files]
        for split in ('RR','FR'):
            for metric in rs[0]['splits'][split]['mean']:
                a,b=[r['splits'][split]['mean'][metric] for r in rs]
                rows.append({'seed':seed,'split':split,'metric':metric,'baseline':a,'ema':b,'ema_minus_baseline':b-a})
            scenes=[read(root/a/(split+'_per_scene.json')) for a in ('baseline','ema')]
            if [r['scene_index'] for r in scenes[0]]!=[r['scene_index'] for r in scenes[1]]:raise ValueError('Scene IDs differ')
            for a,b in zip(*scenes):deltas.append({'seed':seed,'split':split,'scene_index':a['scene_index'],**{k:b[k]-a[k] for k in a if k!='scene_index'}})
    write(output/'paired_scene_deltas.json',deltas)
    if rows:
        with open(output/'comparison.csv','w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        groups={}
        for r in rows:groups.setdefault((r['split'],r['metric']),[]).append(r)
        statistics=[]
        for (split,metric),values in groups.items():
            item={'split':split,'metric':metric,'n_seeds':len(values)}
            for k in ('baseline','ema','ema_minus_baseline'):
                x=[r[k] for r in values];item[k+'_mean']=float(np.mean(x));item[k+'_sample_std']=float(np.std(x,ddof=1)) if len(x)>1 else None
            statistics.append(item)
        write(output/'seed_statistics.json',statistics)
def package(note):
    curves();summarize();write(stage/'status.json',{'note':note,'phase':PHASE,'seed':SEED,'remote_preservation_verified':False})
    inventory={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
    write(output/'artifact_manifest.json',inventory)
    archive=Path('/kaggle/working')/(RUN_TAG+'_'+phasekey+'_results.zip');temp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in list(inventory)+['artifact_manifest.json']:z.write(output/name,RUN_TAG+'/'+name)
    os.replace(temp,archive);print(note,'\nZIP:',archive,'\nSHA-256:',sha(archive),flush=True)
    from IPython.display import display,FileLink
    display(FileLink(str(archive)))
note='paused/failed; existing checkpoint evidence only'
try:
    if PHASE=='smoke':
        jobs=[task('smoke',42,'baseline',0,[],None,2,stop=1),task('smoke',42,'ema',1,[0.99,0.999],0.999,2,stop=1)]
        # Forced optimizer-boundary pause, then restore every EMA/RNG/optimizer state.
        if not complete(jobs):run(jobs)
        if time.time()<deadline:
            for j in jobs:j['config']['stop_after_group']=0;write(j['path'],j['config'])
            run(jobs)
        if complete(jobs):
            pair_check(jobs)
            refs=[task('smoke_reference',42,'baseline',0,[],None,2),task('smoke_reference',42,'ema',1,[0.99,0.999],0.999,2)]
            if not complete(refs):run(refs)
            if complete(refs):
                checks=[]
                for a,b in zip(jobs,refs):
                    ca=torch.load(a['folder']/'latest.pt',map_location='cpu',weights_only=False);cb=torch.load(b['folder']/'latest.pt',map_location='cpu',weights_only=False)
                    checks.append({'label':a['label'],'online_equal':state_hash(ca['model'])==state_hash(cb['model']),
                         'order_equal':ca['order_history']==cb['order_history'],'ema_update_count_equal':ca['ema_state']['updates']==cb['ema_state']['updates'],
                         'ema_shadows_equal':all(state_hash(s)==state_hash(cb['ema_state']['shadows'][k]) for k,s in ca['ema_state']['shadows'].items())})
                passed=all(all(v for k,v in c.items() if k!='label') for c in checks)
                write(output/'smoke_checks.json',{'passed':passed,'checks':checks,'excluded_from_final':True})
                if not passed:raise ValueError('Resume equivalence failed')
                note='Smoke passed; 2 epochs/64 train/32 validation only; not final performance'
    elif PHASE=='probe':
        if not read(output/'smoke_checks.json')['passed']:raise ValueError('Smoke must pass first')
        jobs=[task('probe',42,'baseline',0,[],None,10),task('probe',42,'ema',1,[0.99,0.999],0.999,10)]
        run(jobs)
        if complete(jobs):pair_check(jobs);note='10-epoch beta probe complete; no test used'
    elif PHASE=='select':
        p=output/'probe/seed_42/ema';marker=p/'training_complete.json'
        if not marker.exists() or read(marker)['epochs']!=10:raise ValueError('Finish beta probe first')
        if not read(output/'probe/seed_42/pair_checks.json')['online_weights_exact_equal']:raise ValueError('Pair checks missing')
        h=read(p/'history.json');scores={str(b):min(r['ema_validation'][str(b)]['mse'] for r in h) for b in (0.99,0.999)}
        beta=min((0.99,0.999),key=lambda b:(scores[str(b)],-b))
        seal={'basis':basis,'selected_beta':beta,'candidates':[0.99,0.999],'validation_scores':scores,'test_used':False,
              'selection_scope':'10-epoch validation proxy; not proven optimum for 100 epochs','evidence_sha256':sha(p/'history.json')}
        path=output/'final_settings.json'
        if path.exists() and read(path)!=seal:raise ValueError('Selection already frozen; cannot change')
        write(path,seal);note='beta frozen from validation: '+str(beta)
    else:
        seal=read(output/'final_settings.json')
        if seal['basis']!=basis or seal['test_used'] or seal['evidence_sha256']!=sha(output/'probe/seed_42/ema/history.json'):raise ValueError('Frozen selection evidence changed')
        beta=seal['selected_beta'];test=PHASE=='final_test'
        jobs=[task('final',SEED,'baseline',0,[],None,100,'test' if test else 'train'),task('final',SEED,'ema',1,[beta],beta,100,'test' if test else 'train')]
        if test:
            if not complete(jobs):raise ValueError('Finish both arms for this seed before test')
            pair_check(jobs)
        run(jobs)
        if complete(jobs,test):
            if not test:pair_check(jobs)
            note=phasekey+' complete; '+('one-seed result, not three-seed conclusion' if SEED==42 else 'aggregate includes completed seeds only')
finally:
    package(note)
