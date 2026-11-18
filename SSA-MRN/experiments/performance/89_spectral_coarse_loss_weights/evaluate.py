"""89 evaluation. Each validation-selected run (best.pt = minimum original-scale validation MSE) on the 20 RR test scenes of its sensor,
without and with the 86 K10 projection (`+P`). Controls of the same seed, computed once in `run.py --mode optimize` (GPU idle):
C2MS-SP (65 C2MS_L1S + P) and the 88 K2D (+P). Test is measured and reported only; nothing is selected with it (selection is in
selection.py on validation). FR and the WV2 transfer are not measured in this experiment (screening); the confirmation stage adds them.
measure_out_impl() times the two W2 output implementations (85 official setting and the 86 CUDA graph + FP16 setting, 256^2 and 512^2)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys,time
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
import model as M
from prepare import SENSORS,data_files,FastProjected,Graphed
from train import T,sha,write,now,run_name,set_impl
from common.metrics import rr_metrics


_SETUP=[False]
def setup(seed):
    """58 setup (seeds, deterministic flags, one thread) once per process: torch refuses a second set_num_interop_threads."""
    if not _SETUP[0]:T.setup(seed);_SETUP[0]=True


def predict(net,pan,lms,ms,scale):
    x=[torch.from_numpy((np.asarray(a,dtype=np.float64)/scale).astype('float32')).unsqueeze(0).cuda() for a in (pan,lms,ms)]
    with torch.no_grad():y=net(*x)
    out=y[0].float().cpu().numpy().astype(np.float64)*scale
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite prediction')
    return out


def load(path,arm,bands,seed):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=M.Model(arm,bands,seed);m.load_state_dict(ck['model'],strict=True)
    return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'),metrics=ck.get('metrics'))


def rr_rows(cfg,sensor,models):
    p=Path(cfg['data_root'])/data_files(sensor)['RR']
    if sha(p)!=cfg['data_sha256'][sensor]['RR']:raise ValueError('test data mismatch '+sensor)
    rows={n:[] for n in models};s=SENSORS[sensor]['max_value']
    with torch.inference_mode(),h5py.File(p) as f:
        if len(f['pan'])!=20:raise ValueError('expected 20 RR scenes')
        for i in range(20):
            a={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')}
            for n,m in models.items():rows[n].append(dict(index=i,RR=rr_metrics(a['gt'],predict(m,a['pan'],a['lms'],a['ms'],s))))
    means={n:{k:float(np.mean([x['RR'][k] for x in v])) for k in v[0]['RR']} for n,v in rows.items()}
    return rows,means


def control_paths(cfg,sensor,seed):
    return {'C2MS-SP':('L1S',HERE.parents[2]/cfg['controls']['c2ms_sp'][sensor].replace('<seed>',str(seed))),
            '88 K2D':('K2D',Path(cfg['controls']['k2d_88'][sensor].replace('<seed>',str(seed))))}


def control_rows(cfg,sensor,seed):
    """Controls of (sensor, seed) with and without projection, cached in runtime/controls (computed once, GPU idle)."""
    p=Path(cfg['runtime'])/'controls'/f'{sensor}_s{seed}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    setup(seed);models={};weights={}
    for name,(arm,path) in control_paths(cfg,sensor,seed).items():
        net,weights[name]=load(path,arm,SENSORS[sensor]['bands'],seed);models[name]=net;models[name+' + P']=FastProjected(net,sensor)
    rows,means=rr_rows(cfg,sensor,models);rec=dict(time_kst=now(),sensor=sensor,seed=seed,weights=weights,means=means,rows=rows,projection='86 FastProjectedV2 K10')
    write(p,rec);return rec


def deltas(a,b):return {k:a[k]-b[k] for k in ('PSNR','SAM','ERGAS')}


def evaluate(cfg,sensor,arm,seed):
    setup(seed);set_impl(cfg);dest=Path(cfg['runtime'])/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    net,w=load(dest/'checkpoints/best.pt',arm,SENSORS[sensor]['bands'],seed);ctrl=control_rows(cfg,sensor,seed)
    rows,means=rr_rows(cfg,sensor,{arm:net,arm+' + P':FastProjected(net,sensor)})
    cmp={'P_vs_C2MS_SP':deltas(means[arm+' + P'],ctrl['means']['C2MS-SP + P']),'P_vs_88_K2D':deltas(means[arm+' + P'],ctrl['means']['88 K2D + P']),
         'raw_vs_65_raw':deltas(means[arm],ctrl['means']['C2MS-SP'])}
    done_rec=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))
    write(dest/'evaluation'/'RR_test.json',dict(time_kst=now(),sensor=sensor,arm=arm,seed=seed,weights=w,means=means,rows=rows,controls=ctrl['means'],delta_RR=cmp,
          validation=done_rec['best_val'],best_epoch=done_rec['best_epoch'],note='test measured and reported only; never used for selection'))
    write(done,dict(time_kst=now(),weights=w,delta_RR=cmp,means=means))
    print(json.dumps(dict(time_kst=now(),evaluated=run_name(sensor,arm,seed),delta_RR=cmp)),flush=True)


def macs(net,shapes):
    """conv_macs (Conv2d only) plus the ConvTranspose2d multiply-accumulates."""
    extra=[0]
    def hook(m,i,o):extra[0]+=i[0].numel()//i[0].shape[0]*i[0].shape[0]*m.out_channels*m.kernel_size[0]*m.kernel_size[1]//m.groups
    hs=[m.register_forward_hook(hook) for m in net.modules() if isinstance(m,torch.nn.ConvTranspose2d)]
    try:base=M.conv_macs(net,shapes)
    finally:
        for h in hs:h.remove()
    return base+extra[0]


def measure_out_impl(cfg,rounds=3,warmup=5,repeats=50):
    """Both W2 output implementations of the full 2x2 C2MS (same random weights mapped, so the outputs agree) at RR 256^2 and FR 512^2 of
    test scene 0, in the 85 official setting and the 86 graph_fp16 setting. Returns timings, the max output difference and the choice."""
    out=dict(time_kst=now(),rule=cfg['w2_check']['output_implementation'],settings={})
    torch.manual_seed(0);base=M.to_2x2(M.M54.Model('C2MS',4,42),'convT')
    with torch.no_grad():
        for p in base.core.backbone.branch[-1].parameters():p.copy_(torch.randn_like(p)*.02)
    nets={}
    for impl in ('convT','pixelshuffle'):
        n=M.to_2x2(M.M54.Model('C2MS',4,42),impl);sd={k:v for k,v in base.state_dict().items()}
        lay=n.core.backbone.branch[-1];w=base.core.backbone.branch[-1].weight.detach()  # [128, 64, 2, 2]
        own={k:v for k,v in n.state_dict().items() if not k.startswith('core.backbone.branch.9.')}
        n.load_state_dict({**{k:sd[k] for k in own},**({'core.backbone.branch.9.weight':w,'core.backbone.branch.9.bias':sd['core.backbone.branch.9.bias']} if impl=='convT' else {'core.backbone.branch.9.conv.weight':w.permute(1,2,3,0).reshape(256,128,1,1),'core.backbone.branch.9.bias':sd['core.backbone.branch.9.bias']})},strict=True)
        nets[impl]=n.cuda().eval()
    mv=SENSORS['QB']['max_value'];xs={}
    for split,size in (('RR','256'),('FR','512')):
        with h5py.File(Path(cfg['data_root'])/data_files('QB')[split]) as f:xs[size]=[torch.from_numpy((np.asarray(f[k][0],dtype=np.float64)/mv).astype('float32'))[None].cuda() for k in ('pan','lms','ms')]
    with torch.inference_mode():
        d=max(float((nets['convT'](*x).double()-nets['pixelshuffle'](*x).double()).abs().max()) for x in xs.values())
    out['max_abs_output_difference']=d
    if d>1e-4:raise AssertionError('the two output implementations differ: %g'%d)
    for setting,(det,graph,fp16) in {'official':(True,False,False),'graph_fp16':(False,True,True)}.items():
        torch.backends.cudnn.deterministic=det;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(det,warn_only=True);sys_={k:(Graphed(v) if graph else v) for k,v in nets.items()};res={}
        for size,x in xs.items():
            per={n:[] for n in sys_}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=fp16):
                for m in sys_.values():
                    for _ in range(warmup):m(*x)
                torch.cuda.synchronize()
                for _ in range(rounds):
                    t={n:[] for n in sys_}
                    for _ in range(repeats):
                        for n,m in sys_.items():
                            a=time.perf_counter();m(*x);torch.cuda.synchronize();t[n].append(time.perf_counter()-a)
                    for n in sys_:per[n].append(float(np.median(t[n])))
            res[size]={n:float(np.median(v)) for n,v in per.items()}
            print(json.dumps(dict(time_kst=now(),timing=setting,size=size,ms={n:round(1e3*v,3) for n,v in res[size].items()})),flush=True)
        out['settings'][setting]=res
    faster=lambda a,b:all(out['settings'][s][z][a]<=0.97*out['settings'][s][z][b] for s in out['settings'] for z in out['settings'][s])
    out['choice']='pixelshuffle' if faster('pixelshuffle','convT') else 'convT'
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    return out
