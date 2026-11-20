"""90 evaluation. Each factorized run <v>K (best.pt = minimum original-scale validation MSE) on the 20 RR test scenes of its sensor, without and
with the 86 K10 projection (`+P`), plus FR QNR (20 FR scenes, `+P`, reference). WV3 runs are also applied to WV2 (transfer, 82 per-band
calibration as for every C2MS-SP model on an untrained sensor; RR and FR, reference only). Controls of the same seed, computed once in
`run.py --mode optimize` (GPU idle) and cached: C2MS-SP (65 C2MS_L1S) and 88 K2D, both with and without the projection. Test is measured
and reported only; nothing is selected with it (the variant is selected on validation in run.py)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import importlib.util,json,sys
from pathlib import Path
import numpy as np,torch,h5py
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
sys.path.insert(0,str(HERE))
import model as M
from prepare import SENSORS,data_files,FastProjected
from train import T,sha,write,now,run_name,set_impl
from common.metrics import rr_metrics,fr_metrics
_spec=importlib.util.spec_from_file_location('exp82_prepare_90',HERE.parent/'82_cross_sensor_band_calibration/prepare.py')
C82=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C82)
TRANSFER={'WV3':['WV2']}


class Calibrated(C82.Calibrated):
    """82 calibration ('scene') with its reference ratios moved to the input device (as 88 evaluate.py)."""
    def forward(self,pan,lms,ms):
        self.r_train=self.r_train.to(pan.device);return super().forward(pan,lms,ms)


_SETUP=[False]
def setup(seed):
    if not _SETUP[0]:T.setup(seed);_SETUP[0]=True


def predict(net,pan,lms,ms,scale):
    x=[torch.from_numpy((np.asarray(a,dtype=np.float64)/scale).astype('float32')).unsqueeze(0).cuda() for a in (pan,lms,ms)]
    with torch.no_grad():y=net(*x)
    out=y[0].float().cpu().numpy().astype(np.float64)*scale
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite prediction')
    return out


def load(path,arm,sensor,seed):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['seed']!=seed:raise ValueError('seed mismatch '+str(path))
    m=M.Model(arm,SENSORS[sensor]['bands'],seed,sensor=sensor);m.load_state_dict(ck['model'],strict=True)
    return m.cuda().eval(),dict(path=str(path),sha256=sha(path),best_epoch=ck.get('best_epoch'),metrics=ck.get('metrics'))


def expected(cfg,target,split):return (cfg['data_sha256'][target] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'][target])[split]


def systems(net,sensor,target,name):
    """{name: net, name+' + P': projected}; on a transfer target both are wrapped in the 82 calibration first."""
    if target!=sensor:
        ref=json.loads((HERE.parent/'82_cross_sensor_band_calibration'/'results/reference_stats'/f'{sensor}.json').read_text(encoding='utf8'));net=Calibrated(net,ref,'scene')
    return {name:net,name+' + P':FastProjected(net,target)}


def rows_for(cfg,target,models,fr_for):
    """RR metrics of every model and FR QNR of the models in fr_for on the 20 test scenes of target."""
    rr=Path(cfg['data_root'])/data_files(target)['RR'];fr=Path(cfg['data_root'])/data_files(target)['FR']
    if sha(rr)!=expected(cfg,target,'RR') or sha(fr)!=expected(cfg,target,'FR'):raise ValueError('test data mismatch '+target)
    rows={n:[] for n in models};s=SENSORS[target]['max_value']
    with torch.inference_mode(),h5py.File(rr) as r,h5py.File(fr) as f:
        if len(r['pan'])!=20 or len(f['pan'])!=20:raise ValueError('expected 20 scenes')
        for i in range(20):
            a={k:np.asarray(r[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')};b={k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms')}
            for n,m in models.items():
                row=dict(index=i,RR=rr_metrics(a['gt'],predict(m,a['pan'],a['lms'],a['ms'],s)))
                if n in fr_for:row['FR']=fr_metrics(predict(m,b['pan'],b['lms'],b['ms'],s),b['lms'],b['ms'],b['pan'])
                rows[n].append(row)
    means={n:{p:{k:float(np.mean([x[p][k] for x in v])) for k in v[0][p]} for p in ('RR','FR') if p in v[0]} for n,v in rows.items()}
    return rows,means


def control_paths(cfg,sensor,seed):
    return {'C2MS-SP':('L1S',HERE.parents[2]/cfg['controls']['c2ms_sp'][sensor].replace('<seed>',str(seed))),
            '88 K2D':('K2D',Path(cfg['controls']['k2d_88'][sensor].replace('<seed>',str(seed))))}


def control_rows(cfg,sensor,target,seed):
    """Controls of (sensor, target, seed) with and without projection, cached in runtime/controls (computed once, GPU idle)."""
    p=Path(cfg['runtime'])/'controls'/f'{sensor}_{target}_s{seed}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    setup(seed);models={};weights={}
    for name,(arm,path) in control_paths(cfg,sensor,seed).items():
        net,weights[name]=load(path,arm,sensor,seed);models.update(systems(net,sensor,target,name))
    rows,means=rows_for(cfg,target,models,[n for n in models if n.endswith(' + P')])
    rec=dict(time_kst=now(),sensor=sensor,target=target,seed=seed,weights=weights,means=means,rows=rows,projection='86 FastProjectedV2 K10',
             calibration='82 scene' if target!=sensor else None)
    write(p,rec);return rec


def deltas(a,b):return {k:a[k]-b[k] for k in ('PSNR','SAM','ERGAS')}


def evaluate(cfg,sensor,arm,seed):
    setup(seed);set_impl(cfg);dest=Path(cfg['runtime'])/'runs'/run_name(sensor,arm,seed);done=dest/'evaluation_complete.json'
    if done.exists():return
    if arm not in M.FACTORIZED:raise ValueError('only the factorized final models are evaluated: '+arm)
    if not (dest/'training_complete.json').exists():raise RuntimeError('training incomplete')
    net,w=load(dest/'checkpoints/best.pt',arm,sensor,seed);out={}
    for target in [sensor]+TRANSFER.get(sensor,[]):
        ctrl=control_rows(cfg,sensor,target,seed);models=systems(net,sensor,target,arm)
        rows,means=rows_for(cfg,target,models,[arm+' + P'])
        cmp={'P_vs_88_K2D_P':deltas(means[arm+' + P']['RR'],ctrl['means']['88 K2D + P']['RR']),'P_vs_C2MS_SP':deltas(means[arm+' + P']['RR'],ctrl['means']['C2MS-SP + P']['RR']),
             'raw_vs_88_K2D_raw':deltas(means[arm]['RR'],ctrl['means']['88 K2D']['RR'])}
        write(dest/'evaluation'/f'{target}_test.json',dict(time_kst=now(),sensor=sensor,target=target,arm=arm,seed=seed,weights=w,means=means,rows=rows,
              controls=ctrl['means'],delta_RR=cmp,role='judgement target (report)' if target==sensor else 'transfer, reference only',note='test measured and reported only; never used for selection'))
        out[target]=dict(means=means,delta_RR=cmp)
    done_rec=json.loads((dest/'training_complete.json').read_text(encoding='utf8'))
    write(done,dict(time_kst=now(),weights=w,targets=out,validation=done_rec['best_val'],best_epoch=done_rec['best_epoch']))
    print(json.dumps(dict(time_kst=now(),evaluated=run_name(sensor,arm,seed),delta_RR={t:v['delta_RR'] for t,v in out.items()})),flush=True)


def macs(net,shapes):
    """conv_macs (Conv2d only) plus the ConvTranspose2d multiply-accumulates (fixed resampling filters are not counted; reported apart)."""
    extra=[0]
    def hook(m,i,o):extra[0]+=i[0].numel()//i[0].shape[0]*i[0].shape[0]*m.out_channels*m.kernel_size[0]*m.kernel_size[1]//m.groups
    hs=[m.register_forward_hook(hook) for m in net.modules() if isinstance(m,torch.nn.ConvTranspose2d)]
    try:base=M.conv_macs(net,shapes)
    finally:
        for h in hs:h.remove()
    return base+extra[0]
