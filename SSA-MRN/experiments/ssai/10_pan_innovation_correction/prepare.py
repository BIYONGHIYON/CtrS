"""Train-input response fit, validation-only diagnosis, frozen fresh-run plan."""
import argparse, importlib, json, math, shutil, sys, time, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
module=importlib.import_module('experiments.ssai.10_pan_innovation_correction.model')
helpers=importlib.import_module('experiments.ssai.10_pan_innovation_correction.train')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    a=parser.parse_args()
    if a.runtime.exists(): raise FileExistsError('Fresh runtime required')
    if shutil.disk_usage(a.previous).free<20*1024**3: raise RuntimeError('Need20GiB free')
    old=json.loads((a.previous/'plan.json').read_text())
    actual={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)}
    for k,v in actual.items():
        if old['environment'][k]!=v: raise ValueError('Environment changed: '+k)
    for name in ('common/model.py','references/upstream/network.py'):
        if suite.sha(ROOT/name)!=old['source_hashes'][name]: raise ValueError('Original architecture changed')
    cache=Path(old.get('cache_root',a.previous/'cache'))
    print('phase: data and checkpoint verification',flush=True)
    for split in ('train','val'):
        info=old['data'][split]
        if suite.sha(info['path'])!=info['sha256']: raise ValueError('Original data changed')
        for key,h in info['cache'].items():
            if suite.sha(cache/split/(key+'.npy'))!=h: raise ValueError('Cache changed')
    cp=a.previous/'A0/best.pt'
    cp_hash=suite.sha(cp)
    if cp_hash!=json.loads((a.previous/'A0/complete.json').read_text())['best_sha256']: raise ValueError('A0 weight changed')
    state=torch.load(cp,map_location='cpu',weights_only=False)
    suite.configure(42,'cuda'); control=RestoredPansharpeningNet(4,4)
    initial={k:v.clone() for k,v in control.state_dict().items()}
    init_hash=helpers.state_hash(initial)
    # Compare with the preserved, previously verified actual server initialization.
    expected='c24fee837e68d93a796b47949a4b7dfde0cf1860734532ea2b39cbf2ded56ea6'
    if init_hash!=expected: raise ValueError('Not experiment4 A0 common initial state')
    tr,va=(suite.Cached(cache/s) for s in ('train','val'))
    print('phase: fixed PAN response fit from512 train INPUTS only',flush=True)
    # 512 evenly spaced train indices; no validation/test/GT used for fitting.
    fit_indices=np.linspace(0,len(tr)-1,512,dtype=int).tolist()
    xx=torch.zeros(5,5,dtype=torch.float64); xy=torch.zeros(5,dtype=torch.float64)
    for batch in DataLoader(Subset(tr,fit_indices),batch_size=4,shuffle=False,num_workers=0):
        pan,lms=batch[0].cuda(),batch[1].cuda()
        x=module.lowpass(lms)[:,:,::8,::8].permute(0,2,3,1).reshape(-1,4).cpu().double()
        y=module.lowpass(pan)[:,:,::8,::8].flatten().cpu().double()
        x=torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
        xx+=x.T@x; xy+=x.T@y
    ridge=1e-6*xx.diag()[:4].mean()
    regularizer=torch.diag(torch.tensor([1.,1.,1.,1.,0.],dtype=torch.float64))*ridge
    response=torch.linalg.solve(xx+regularizer,xy).float().tolist()
    if not all(math.isfinite(v) for v in response): raise ValueError('Nonfinite response')
    print('phase: finite/shape/common-initialization/gradient/zero-innovation preflight',flush=True)
    checks={'passed':False,'initial_common_sha256':init_hash,'models':{}}
    sample=[v[:].unsqueeze(0).cuda() for v in va[0][:3]]
    gt=va[0][3].unsqueeze(0).cuda()
    for arm in ('A1','A2'):
        suite.configure(42,'cuda'); net=module.PanInnovationNet(innovation=arm=='A2').cuda(); net.set_response(response)
        assert helpers.state_hash(helpers.common_state(net))==init_hash
        F.mse_loss(net(*sample),gt).backward()
        grads={name:float(p.grad.detach().norm()) for name,p in net.named_parameters() if p.grad is not None and name.startswith(module.STAGES)}
        assert all(math.isfinite(v) for v in grads.values())
        for stage in module.STAGES:
            for i in range(4):
                for suffix in ('conv1t6.weight','conv6t6.weight','conv7t6_3.weight','gamma'):
                    assert grads[f'{stage}.{i}.{suffix}']>0
        net.zero_grad(set_to_none=True)
        with torch.inference_mode():
            shapes=[]
            for h,w in ((64,64),(256,256),(512,512),(32,48)):
                inputs=[torch.rand(1,1,h,w,device='cuda'),torch.rand(1,4,h,w,device='cuda'),torch.rand(1,4,h//4,w//4,device='cuda')]
                y=net(*inputs); assert y.shape==(1,4,h,w) and torch.isfinite(y).all()
                shapes.append([h,w])
            if arm=='A2':
                for stage in module.STAGES:
                    b=getattr(net,stage)[0]; x=torch.rand(1,5,16,24,device='cuda')
                    pan=(x[:,:4]*b.pan_response[:4].view(1,4,1,1)).sum(1,keepdim=True)+b.pan_response[4]
                    assert torch.count_nonzero(b(pan,x))==0
        checks['models'][arm]={'registered_parameters':sum(p.numel() for p in net.parameters()),'ssai_parameters_with_gradients':sum(p.numel() for n,p in net.named_parameters() if n in grads),'gradient_norms':grads,'shapes':shapes,'common_initial_match':True,'zero_innovation_zero_output':arm=='A2'}
        del net
    assert checks['models']['A1']['registered_parameters']==checks['models']['A2']['registered_parameters']
    print('phase: A0 validation128 directional diagnosis (no parameter update)',flush=True)
    control.load_state_dict(state['model'],strict=True);control.cuda().eval()
    # Full/high-frequency PAN error against the current stage's MS error;
    # use GT only for this recorded validation diagnostic, never candidate fitting.
    captured={};handles=[]
    for stage in module.STAGES:
        def hook(m,args,output,s=stage): captured[s]=tuple(x.detach() for x in args)
        handles.append(getattr(control,stage)[0].register_forward_hook(hook))
    stats={s:{kind:[0.,0.,0.] for kind in ('direct','innovation')} for s in module.STAGES}
    val_indices=list(range(128))
    with torch.inference_mode():
        for batch in DataLoader(Subset(va,val_indices),batch_size=4,shuffle=False,num_workers=0):
            pan,lms,ms,gt=(x.cuda() for x in batch); control(pan,lms,ms)
            for stage in module.STAGES:
                p,features=captured[stage]; current=features[:,:4]
                target=F.interpolate(gt,size=current.shape[-2:],mode='bilinear',align_corners=False)
                error=module.highpass(target-current)
                predicted=(current*torch.tensor(response[:4],device='cuda').view(1,4,1,1)).sum(1,keepdim=True)+response[4]
                for kind,d in (('direct',module.highpass(p)),('innovation',module.highpass(p-predicted))):
                    d=d.expand_as(error); values=stats[stage][kind]
                    values[0]+=(d.double()*error.double()).sum().item()
                    values[1]+=d.double().square().sum().item()
                    values[2]+=error.double().square().sum().item()
    for h in handles:h.remove()
    diagnosis={'validation_indices':val_indices,'baseline_best':{'path':str(cp),'sha256':cp_hash,'epoch':state['epoch']},'fit':{'split':'train','indices':fit_indices,'uses_GT':False,'coefficients':response,'ridge':float(ridge),'definition':'fixed ridge affine response; unconstrained coefficients; not exact sensor model'},'stages':{}}
    for stage in module.STAGES:
        diagnosis['stages'][stage]={}
        for kind,(dot,d2,e2) in stats[stage].items():
            diagnosis['stages'][stage][kind]={'pooled_cosine':dot/math.sqrt(max(d2*e2,1e-30)),'positive_direction':dot>0,'diagnostic_best_scalar':dot/max(d2,1e-30),'diagnostic_only_error_energy_reduction':dot*dot/max(d2*e2,1e-30)}
    # A deliberately weak necessary gate: positive signed error alignment at all
    # stages. This is no promise of final decoder accuracy or candidate advantage.
    diagnosis['direction_gate_passed']=all(v['innovation']['positive_direction'] for v in diagnosis['stages'].values())
    diagnosis['limitation']='Bilinearly resized validation GT is a stage diagnostic, not extra supervision. Candidates use only original final MSE.'
    a.runtime.mkdir(parents=True)
    suite.write(a.runtime/'diagnostics.json',diagnosis)
    checks['passed']=True; suite.write(a.runtime/'preflight.json',checks)
    first=json.loads((a.previous/'A0/history.json').read_text())[0]['val_mse']
    p={k:v for k,v in old.items() if k not in ('initial_checkpoints','optimization_benchmark','vram_safety','reference_area_candidate')}
    p.update(arms=['A1','A2'],experiment=10,cache_root=str(cache),cache_mode='gpu_partial',cache_fraction=.85,initial_common_sha256=init_hash,pan_response=response,environment=actual,created_utc=time.time(),baseline_reference={'path':str(cp),'sha256':cp_hash},divergence={'first_epoch_mse_limit':2*first,'reference':'2*A0 epoch1 validation','automatic_restart':False},followup={'automatic_launch':False},source_hashes={})
    paths=list((ROOT/'common').glob('*.py'))+[ROOT/'references/upstream/network.py']+list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'config.json']
    for path in paths:
        name=path.relative_to(ROOT);p['source_hashes'][name.as_posix()]=suite.sha(path)
        dest=a.runtime/'provenance/source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    suite.write(a.runtime/'plan.json',p);suite.write(a.runtime/'provenance/plan.json',p)
    suite.save(a.runtime/'provenance/initial_state.pt',initial)
    with zipfile.ZipFile(a.runtime/'provenance/training_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in (a.runtime/'provenance').rglob('*'):
            if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(a.runtime/'provenance'))
    print(json.dumps({'preflight':checks,'direction_gate':diagnosis['direction_gate_passed'],'diagnostic_stages':diagnosis['stages'],'pan_response':response}),flush=True)
    if not diagnosis['direction_gate_passed']:raise RuntimeError('No positive innovation alignment at every stage; full training refused')


if __name__=='__main__': main()
