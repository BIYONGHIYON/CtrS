"""Explicit adapters for retained trusted models, isolated per checkpoint process."""
import ast,base64,hashlib,importlib,importlib.util,json,sys,types,zlib
from pathlib import Path
import torch

def module(path,name='audit_model'):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def bundle(folder,runtime):
    """Materialize only syntactically valid embedded Python sources; never run controllers."""
    out=runtime/'adapter_sources'/folder.name;out.mkdir(parents=True,exist_ok=True)
    for source in [folder/'run.py',*sorted((folder/'provenance').glob('**/run_cell.py'))]:
        if not source.exists():continue
        tree=ast.parse(source.read_text(encoding='utf8'))
        literals={}
        for n in tree.body:
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name):
                        try:literals[target.id]=ast.literal_eval(n.value)
                        except (ValueError,TypeError):pass
        if 'PACKED_SOURCES' in literals:
            packed=base64.b64decode(literals['PACKED_SOURCES']);assert hashlib.sha256(packed).hexdigest()==literals['PAYLOAD_SHA256']
            decoded=json.loads(zlib.decompress(packed).decode('utf8'))
            for k,v in decoded.items():
                assert hashlib.sha256(v.encode()).hexdigest()==literals['EMBEDDED_HASHES'][k]
                dest=out/k;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v.encode())
        for n in tree.body:
            if not isinstance(n,ast.Assign):continue
            try:d=ast.literal_eval(n.value)
            except (ValueError,TypeError):continue
            if not isinstance(d,dict):continue
            for k,v in d.items():
                if not isinstance(k,str) or not k.endswith(('.py','.json')) or not isinstance(v,str) or len(v)<100:continue
                try:
                    if k.endswith('.py'):ast.parse(v)
                    else:json.loads(v)
                except (SyntaxError,ValueError):continue
                dest=out/k
                if not dest.resolve().is_relative_to(out.resolve()):raise ValueError('Source outside bundle')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v.encode('utf8'))
    sys.path.insert(0,str(out));return out

def load(root,folder,cp):
    sys.path.insert(0,str(root/'SSA-MRN'))
    from common import checkpoint as legacy
    from common.model import RestoredPansharpeningNet
    relative=folder.relative_to(root);track=relative.parts[2];n=int(folder.name.split('_')[0]);legacy.RESEARCH_TRACK=track
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=ck.get('config',{});plan=cfg.get('plan',{});flat={**plan,**cfg}
    arm=ck.get('arm',cfg.get('arm',cfg.get('variant',ck.get('variant',cp.parent.name.split('_')[0]))))
    sensor=flat.get('sensor',ck.get('sensor','QB'));channels=flat.get('channels',8 if sensor in ('WV2','WV3') else 4);k=flat.get('k',ck.get('ssai_dimension',4))
    calibration=plan.get('calibration',cfg.get('calibration'))
    if track=='ssai' and n==10:net=module(folder/'model.py').LocalWindowSSAINet(arm,channels,k)
    elif track=='ssai' and n==12:
        stats=json.loads((folder/'results/evaluation/summary.json').read_text(encoding='utf8'))['observation_statistics']
        net=module(folder/'model.py').ObservationMatchedNet(channels,k,local=arm=='B2',eps=stats['eps'],gain_max=stats['gain_max'])
    elif track=='variable_scale':
        source=folder
        if n==2:source=next((root/'SSA-MRN/experiments/variable_scale').glob('1_*'))
        out=bundle(source,root);
        if not (out/'engine.py').exists():
            source=next((root/'SSA-MRN/experiments/variable_scale').glob('1_*'));out=bundle(source,root)
        pkg='audit_scale_package';m=types.ModuleType(pkg);m.__path__=[str(out)];sys.modules[pkg]=m
        engine=importlib.import_module(pkg+'.engine');modelmod=importlib.import_module(pkg+'.model');datamod=importlib.import_module(pkg+'.data')
        legacy_arm=cp.parent.name.startswith('A_');conditioned=cp.parent.name.startswith('D_');job=dict(legacy=legacy_arm,conditioned=conditioned)
        net=engine.make_model(job,flat.get('seed',42),'cpu')
        net._audit_scale_data=datamod;net._audit_legacy=legacy_arm
    elif track=='performance' and n in (43,44):net=module(folder/'model.py').Model(arm,ck.get('seed',42))
    elif track=='performance' and n in (26,30):net=RestoredPansharpeningNet(4,4)
    elif track=='performance' and n in (24,27):
        path=folder/'provenance/code';sys.path.insert(0,str(path/'SSA-MRN/src'))
        net=module(path/('alpha_models.py' if n==24 else 'controlled_models.py')).build(ck['arm'])
    elif track=='performance' and n==28:net=module(folder/'model.py').RefinementNet(arm,channels)
    elif track=='performance' and n==29:net=module(folder/'model.py').StructureNet(arm,channels)
    elif track=='performance' and n==31:
        if arm=='G5':net=legacy.model_module(28).RefinementNet('G5',4)
        else:
            out=bundle(folder,root);src=out/'worker.py';tree=ast.parse(src.read_text(encoding='utf8'));tree.body=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.ClassDef))]
            # Keep just imports and Model declaration, never invoke a training worker.
            ns={'__file__':str(src),'__name__':'audit_frozen_worker'};exec(compile(tree,str(src),'exec'),ns)
            if calibration is None:
                cfgfile=json.loads((folder/'provenance/config.json').read_text(encoding='utf8'));calibration=cfgfile.get('calibration')
            net=ns['Model'](arm,calibration)
    elif track=='performance' and n==32:
        eps=None
        for p in folder.rglob('*.json'):
            if 'provenance' in p.parts:continue
            try:d=json.loads(p.read_text(encoding='utf8'))
            except (ValueError,UnicodeError):continue
            if isinstance(d,dict) and 'eps' in d and isinstance(d['eps'],(int,float)):eps=d['eps'];break
            if isinstance(d,dict) and 'D4' in d and isinstance(d['D4'],dict):eps=d['D4'].get('selected_eps',eps)
        if eps is None and arm in ('W2',):raise ValueError('Frozen guided-prior eps absent')
        net=module(folder/'model.py').ScreenNet(arm,eps or 1e-4)
    elif track=='performance' and n in (33,38,40):
        source=folder if n!=38 else next((root/'SSA-MRN/experiments/performance').glob('33_*'))
        out=bundle(source,root);net=module(out/'normalization.py').Model(arm,cfg.get('std_floor',.01))
    elif track=='performance' and n==34:net=module(folder/'model.py').PaperSSAMRN(channels,k)
    elif track=='performance' and n in (37,39,41,42):
        out=bundle(folder,root);name={37:'gate29_model',39:'localnorm_model',41:'photoaug_model',42:'dilated_model'}[n];m=module(out/(name+'.py'))
        net=m.W2Net(cfg['gated']) if n==37 else m.Model(cfg.get('arm','baseline'),cfg.get('std_floor',.01),cfg.get('window',5)) if n==39 else m.Model(cfg['arm'])
    else:return (*legacy.load_model(n,cp),dict(track=track,number=n,sensor=sensor))
    if calibration is not None and hasattr(net,'configure'):net.configure(calibration)
    # Observation kernels are registered with placeholder shapes in the original constructors.
    # Restore the exact checkpoint buffer shape before strict loading; no numerical substitution.
    for name,value in ck['model'].items():
        if name.endswith('blur_kernel'):
            parent,_,leaf=name.rpartition('.');owner=net.get_submodule(parent) if parent else net
            if leaf in owner._buffers:owner._buffers[leaf]=torch.empty_like(value)
    net.load_state_dict(ck['model'],strict=True);net.eval()
    return net,ck,dict(track=track,number=n,sensor=sensor)

def predict(net,context,pan,lms,ms,ratio=4):
    if context['track']=='variable_scale' and not net._audit_legacy:return net(pan,lms,ms,ratio)
    if context['track']=='performance' and context['number']==11:return net(pan,ms)
    return net(pan,lms,ms)
