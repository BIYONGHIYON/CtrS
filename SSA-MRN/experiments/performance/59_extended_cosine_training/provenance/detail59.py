"""Sharpness, detail fidelity and colour bias of C2MS 150 (54) vs 300 (59) epochs, seed 42, WV3-trained, on all 20 RR scenes of WV3 and WV2.
Same definitions as 54 evaluate.transfer_detail (sharpness = mean |Laplacian| output / GT; detail = corr and std ratio of (output - LMS) to (GT - LMS);
bias = per-band (mean output - mean GT) / mean GT). Run after training; test never used for selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='provenance' else Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]));sys.path.insert(0,str(HERE))
import numpy as np,torch,h5py
from evaluate import V,Model,SENSORS,data_files,sha
cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else Path(cfg['runtime'])
paths={'C2MS_150':HERE.parents[2]/cfg['controls']['WV3'],'C2MS_300':runtime/'runs/WV3_C2MS_s42/checkpoints/best.pt'}
lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],dtype=torch.float64).view(1,1,3,3)
def hf(x):t=torch.from_numpy(x)[:,None];return float(torch.nn.functional.conv2d(torch.nn.functional.pad(t,(1,1,1,1),mode='replicate'),lap).abs().mean())
nets={}
for k,p in paths.items():
    ck=torch.load(p,map_location='cpu',weights_only=False);n=Model('C2MS',8,42);n.load_state_dict(ck['model'],strict=True);nets[k]=(n.cuda().eval(),sha(p),ck['best_epoch'])
res={}
for target in ('WV3','WV2'):
    rr=Path(cfg['data_root'])/data_files(target)['RR']
    exp=(cfg['data_sha256'] if target in cfg['data_sha256'] else cfg['transfer_data_sha256'])[target]['RR']
    if sha(rr)!=exp:raise ValueError('data sha mismatch')
    with h5py.File(rr) as f:scenes=[{k:np.asarray(f[k][i],dtype=np.float64) for k in ('pan','lms','ms','gt')} for i in range(len(f['pan']))]
    assert len(scenes)==20
    res[target]={}
    for k,(n,h,e) in nets.items():
        rows=[]
        for a in scenes:
            p_=V.predict(n,a['pan'],a['lms'],a['ms'],2047.,'cuda');g=a['gt'];dp,dg=(p_-a['lms']).ravel(),(g-a['lms']).ravel()
            rows.append(dict(sharpness=hf(p_)/hf(g),detail_corr=float(np.corrcoef(dp,dg)[0,1]),detail_gain=float(dp.std()/dg.std()),bias=((p_-g).mean(axis=(1,2))/g.mean(axis=(1,2))).tolist()))
        m={x:float(np.mean([r[x] for r in rows])) for x in ('sharpness','detail_corr','detail_gain')}
        m['band_bias']=np.mean([r['bias'] for r in rows],axis=0).tolist();m['max_abs_band_bias']=float(np.max(np.abs(m['band_bias'])))
        m['weights_sha256']=h;m['best_epoch']=e;res[target][k]=m
        print(target,k,{x:(round(v,4) if isinstance(v,float) else v) for x,v in m.items() if x!='weights_sha256'},flush=True)
out=HERE/'results/transfer_detail.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(description=__doc__,bands='WV3/WV2 band order coastal, blue, green, yellow, red, red edge, NIR1, NIR2; display RGB = bands [4,2,1]',results=res,script_sha256=sha(__file__)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
