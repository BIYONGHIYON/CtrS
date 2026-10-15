import os,sys,json,time,hashlib,shutil,subprocess,zipfile,signal,math,csv
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert PHASE in ('calibrate','smoke','sweep','select','final_train','final_test')
assert SWEEP_SESSION in (1,2)
assert SEED in (42,43,44) and 0<MAX_SESSION_HOURS<=6
assert RUN_TAG.startswith('FFL_1st_') and all(c.isalnum() or c in '_-' for c in RUN_TAG)
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
    if e['name']!='FFL_1st' or e['tag']!=RUN_TAG:raise ValueError('Different experiment; nothing copied')
    m=read(root/'artifact_manifest.json')
    for name,v in m.items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=v['sha256']:raise ValueError('Archive evidence mismatch: '+name)
    return e
if not output.exists() and PHASE!='calibrate':
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
                   if read(p).get('name')=='FFL_1st' and read(p).get('tag')==RUN_TAG]
            if len(roots)!=1:raise ValueError('Set RESUME_FROM to exact attached folder containing experiment.json. Attach ONLY latest cumulative output.')
            verify(roots[0]);shutil.copytree(roots[0],output)
if output.exists() and not (output/'experiment.json').exists():
    raise FileExistsError('Existing directory is not a valid FFL experiment. Use a new RUN_TAG; nothing overwritten.')
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
from ffl_dashboard import Dashboard
from ffl_worker import state_hash
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
dashboard=Dashboard(PHASE,output)
TRAIN_SEED=SEED;SEED=42
DATA_ROOT='/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset'
MODE='smoke' if PHASE in ('calibrate','smoke') else 'full'
exec(SOURCES['ffl_preflight.py'])
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
protocol={'name':'FFL_1st','tag':RUN_TAG,'baseline_audit':BASELINE_AUDIT,'sources':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},
    'data':raw,'k':4,'micro':32,'effective_batch':32,'epochs':100,'lr':1e-4,'optimizer':'Adam defaults','precision':'FP32; AMP/TF32 off; deterministic',
    'loss':'MSE + lambda * FFL; no EMA','ffl':{'alpha':1,'patch_factor':1,'fft_norm':'ortho','batch_matrix':False,'ave_spectrum':False,'log_matrix':False,'weights_detached':True,'normalization':'per sample/band spatial maximum; zero maximum -> zero weights'},
    'lambda_factors':[0.01,0.1,1.0],'candidate_rule':'first64 TRAIN initial MSE/FFL ratio; 8 significant digits; immutable before sweep',
    'selection_rule':'candidate with lowest validation-MSE-best score; tie smaller lambda',
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
phasekey=('sweep_session_'+str(SWEEP_SESSION)) if PHASE=='sweep' else PHASE+('_seed_'+str(SEED) if PHASE.startswith('final_') else '')
stage=output/'sessions'/phasekey;stage.mkdir(parents=True,exist_ok=True)
def task(family,seed,label,gpu,weight,epochs,job_type='train',stop=0):
    folder=output/family/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    initial=folder/'initial.pt'
    if not initial.exists():
        if job_type=='test' or any(folder.rglob('latest.pt')):raise ValueError('Saved initial state missing')
        torch.manual_seed(seed);m=RestoredPansharpeningNet(channels=4,ssai_dimension=4)
        torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},initial)
        (folder/'initial.sha256').write_text(sha(initial));del m
    if sha(initial)!=(folder/'initial.sha256').read_text():raise ValueError('Initial hash differs')
    if seed==42 and state_hash(torch.load(initial,map_location='cpu',weights_only=True))!=catalogue['initial_state_sha256']:
        raise ValueError('Seed42 initialization differs from candidate calibration')
    cond={'basis':basis,'family':family,'seed':seed,'label':label,'epochs':epochs,'lambda':weight,'catalogue_sha256':catalogue_hash,
          'initial_sha256':sha(initial),'seal_sha256':sha(output/'final_settings.json') if family=='final' else None,
          'samples':{s:data[s]['used_samples'] for s in ('train','val')}}
    cfg={'output':str(folder),'code_root':str(code),'conditions':cond,'fingerprint':digest(cond),'data':data,'selection':selection,'seed':seed,'epochs':epochs,
         'batch':32,'micro':32,'amp':False,'lr':1e-4,'deadline':deadline,'reserve_gib':4,'log_seconds':20,'weight':weight,
         'job_type':job_type,'stop_after_group':stop}
    path=folder/(label+'_config.json')
    if path.exists() and read(path)['fingerprint']!=cfg['fingerprint']:raise ValueError('Job config mismatch')
    write(path,cfg)
    return {'gpu':gpu,'label':label,'folder':folder/label,'config':cfg,'path':path}
def run(jobs):
    processes=[]
    try:
        for j in jobs:
            j['folder'].mkdir(exist_ok=True);dashboard.starting(j,0)
            dashboard.rows[j['gpu']]['context']=f"{PHASE} · λ={j['config']['weight']} · {j['folder'].parent.name}"
            log=open(j['folder']/'worker.log','a',encoding='utf-8');environ=os.environ.copy();environ['CUDA_VISIBLE_DEVICES']=str(j['gpu'])
            p=subprocess.Popen([sys.executable,str(code/'ffl_worker.py'),'--config',str(j['path']),'--label',j['label'],'--gpu',str(j['gpu'])],env=environ,stdout=log,stderr=log)
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
    if len({read(p/'initialization.json')['state_sha256'] for p in paths})!=1:raise ValueError('Initial weights differ')
    orders=[read(p/'orders.json') for p in paths]
    if any(o!=orders[0] for o in orders):raise ValueError('Batch orders differ')
    ck=[torch.load(p/'latest.pt',map_location='cpu',weights_only=False) for p in paths]
    if len({(c['epoch'],c['next_sample_position']) for c in ck})!=1:raise ValueError('Incomplete comparison')
    write(paths[0].parent/('pair_checks_'+phasekey+'.json'),{'initial_equal':True,'order_equal':True,
          'labels':[j['label'] for j in jobs],'online_weights_expected_equal':False,'reason':'different training losses'})
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
        root=output/'final'/f'seed_{seed}';files=[root/a/'results.json' for a in ('baseline','ffl')]
        if not all(p.exists() for p in files):continue
        rs=[read(p) for p in files]
        for split in ('RR','FR'):
            for metric in rs[0]['splits'][split]['mean']:
                a,b=[r['splits'][split]['mean'][metric] for r in rs]
                rows.append({'seed':seed,'split':split,'metric':metric,'baseline':a,'ffl':b,'ffl_minus_baseline':b-a})
            scenes=[read(root/a/(split+'_per_scene.json')) for a in ('baseline','ffl')]
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
            for k in ('baseline','ffl','ffl_minus_baseline'):
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
catalogue_path=output/'lambda_catalogue.json';stamp=output/'lambda_catalogue.sha256'
try:
    if PHASE=='calibrate':
        if catalogue_path.exists():
            if read(catalogue_path)['basis']!=basis or not stamp.exists() or stamp.read_text().strip()!=sha(catalogue_path):raise ValueError('Frozen catalogue differs')
            note='Existing frozen lambda candidates reused; not recalibrated'
        else:
            from ffl_loss import ffl_mean
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.manual_seed(42);model=RestoredPansharpeningNet(channels=4,ssai_dimension=4).cuda().eval()
            initial_hash=state_hash(model.state_dict());mse_total=0.;ffl_total=0.;count=0
            with h5py.File(data['train']['path'],'r') as h,torch.inference_mode():
                for start in (0,32):
                    b={k:torch.from_numpy(np.asarray(h[k][start:start+32],dtype=np.float32)).cuda().div_(2047.) for k in ('pan','lms','ms','gt')}
                    pred=model(b['pan'],b['lms'],b['ms']).float()
                    mse_total+=float(torch.nn.functional.mse_loss(pred,b['gt']))*len(pred)
                    ffl_total+=float(ffl_mean(pred,b['gt']))*len(pred);count+=len(pred)
            mse=mse_total/count;ffl=ffl_total/count
            if not (math.isfinite(mse) and math.isfinite(ffl) and mse>0 and ffl>0):raise ValueError('Cannot calibrate FFL lambda from initial loss')
            lambdas=[float(format(f*mse/ffl,'.8g')) for f in (0.01,0.1,1.0)]
            if len(set(lambdas))!=3 or not all(math.isfinite(w) and w>0 for w in lambdas):raise ValueError('Invalid lambda candidates')
            write(catalogue_path,{'basis':basis,'lambdas':lambdas,'factors':[0.01,0.1,1.0],'initial_mse':mse,'initial_ffl':ffl,
                'initial_state_sha256':initial_hash,'data_sha256':data['train']['sha256'],'indices':list(range(64)),
                'normalization':'raw DN /2047; no clipping','alpha':1,'rule':'factor * MSE / FFL at initial weights; training data only','test_used':False})
            stamp.write_text(sha(catalogue_path));del model;torch.cuda.empty_cache()
            note='TRAIN-only calibrated candidates frozen: '+str(lambdas)
    else:
        if not catalogue_path.exists() or not stamp.exists() or stamp.read_text().strip()!=sha(catalogue_path):raise ValueError('Run candidate calibration first; preserve latest Output')
        catalogue=read(catalogue_path);catalogue_hash=sha(catalogue_path)
        if catalogue['basis']!=basis or catalogue['test_used']:raise ValueError('Catalogue protocol differs')
        lambdas=catalogue['lambdas']
        if PHASE=='smoke':
            jobs=[task('smoke',42,'baseline',0,0.,2,stop=1),task('smoke',42,'ffl',1,lambdas[0],2,stop=1)]
            if not complete(jobs):run(jobs)
            if time.time()<deadline:
                for j in jobs:j['config']['stop_after_group']=0;write(j['path'],j['config'])
                run(jobs)
            if complete(jobs):
                pair_check(jobs)
                refs=[task('smoke_reference',42,'baseline',0,0.,2),task('smoke_reference',42,'ffl',1,lambdas[0],2)]
                if not complete(refs):run(refs)
                if complete(refs):
                    checks=[]
                    for a,b in zip(jobs,refs):
                        ca=torch.load(a['folder']/'latest.pt',map_location='cpu',weights_only=False);cb=torch.load(b['folder']/'latest.pt',map_location='cpu',weights_only=False)
                        checks.append({'label':a['label'],'weights_equal':state_hash(ca['model'])==state_hash(cb['model']),
                            'order_equal':ca['order_history']==cb['order_history'],
                            'loss_accumulators_equal':all(torch.equal(ca[k],cb[k]) for k in ('loss_sse','ffl_total','ffl_count','objective_total')),
                            'best_equal':state_hash(ca['best_payload']['model'])==state_hash(cb['best_payload']['model'])})
                    passed=all(all(v for k,v in c.items() if k!='label') for c in checks)
                    write(output/'smoke_checks.json',{'passed':passed,'checks':checks,'excluded_from_final':True})
                    if not passed:raise ValueError('Resume equivalence failed')
                    note='Smoke passed; 2epoch subset only; no test scoring'
        elif PHASE=='sweep':
            if not read(output/'smoke_checks.json')['passed']:raise ValueError('Smoke must pass')
            if SWEEP_SESSION==1:
                jobs=[task('sweep',42,'baseline',0,0.,100),task('sweep',42,'lambda_1',1,lambdas[0],100)]
            else:
                if not all((output/'sweep/seed_42'/a/'training_complete.json').exists() for a in ('baseline','lambda_1')):raise ValueError('Complete sweep session1 first')
                jobs=[task('sweep',42,'lambda_2',0,lambdas[1],100),task('sweep',42,'lambda_3',1,lambdas[2],100)]
            run(jobs)
            if complete(jobs):pair_check(jobs);note=phasekey+' complete; validation only'
        elif PHASE=='select':
            labels=['baseline','lambda_1','lambda_2','lambda_3'];folder=output/'sweep/seed_42'
            evidence={};scores={};initials=[];orders=[]
            for label in labels:
                p=folder/label
                if not (p/'training_complete.json').exists() or read(p/'training_complete.json')['epochs']!=100:raise ValueError('Complete all four sweep jobs')
                cfg=read(folder/(label+'_config.json'));ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
                history=read(p/'history.json');assert len(history)==100 and ck['fingerprint']==cfg['fingerprint']
                assert ck['mse']==min(r['val_mse'] for r in history)
                scores[label]=ck['mse'];initials.append(read(p/'initialization.json')['state_sha256']);orders.append(read(p/'orders.json'))
                for name in ('best.pt','latest.pt','history.json','validation_result.json','training_complete.json'):
                    evidence[str((p/name).relative_to(output))]=sha(p/name)
            if len(set(initials))!=1 or any(o!=orders[0] for o in orders):raise ValueError('Sweep control mismatch')
            chosen=min(range(3),key=lambda i:(scores['lambda_'+str(i+1)],lambdas[i]))
            seal={'basis':basis,'catalogue_sha256':catalogue_hash,'chosen_lambda':lambdas[chosen],'chosen_label':'lambda_'+str(chosen+1),
                  'validation_best_mse':scores,'evidence_sha256':evidence,'test_used':False,'rule':'lowest validation MSE; exact tie smaller lambda',
                  'candidate_better_than_baseline':scores['lambda_'+str(chosen+1)]<scores['baseline']}
            p=output/'final_settings.json'
            if p.exists() and read(p)!=seal:raise ValueError('Final settings already frozen')
            write(p,seal);note='Lambda frozen from validation: '+str(seal['chosen_lambda'])
        else:
            seal=read(output/'final_settings.json')
            if seal['basis']!=basis or seal['catalogue_sha256']!=catalogue_hash or seal['test_used']:raise ValueError('Frozen settings mismatch')
            if any(sha(output/name)!=value for name,value in seal['evidence_sha256'].items()):raise ValueError('Selection evidence changed')
            test=PHASE=='final_test';weight=seal['chosen_lambda']
            jobs=[task('final',SEED,'baseline',0,0.,100,'test' if test else 'train'),task('final',SEED,'ffl',1,weight,100,'test' if test else 'train')]
            if test:
                if not complete(jobs):raise ValueError('Complete BOTH arms for this seed before its test')
                pair_check(jobs)
            run(jobs)
            if complete(jobs,test):
                if not test:pair_check(jobs)
                note=phasekey+' complete; final statistics include completed seeds only'
finally:
    package(note)
