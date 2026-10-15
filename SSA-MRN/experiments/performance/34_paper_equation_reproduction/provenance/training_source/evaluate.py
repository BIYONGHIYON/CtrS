"""Frozen validation-best RR20/FR20 evaluation, no test-based selection."""
import json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from model import PaperSSAMRN, resize
from train import sha, write
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from common.metrics import rr_metrics,fr_metrics
from common.preview import panels


def run(a):
    torch.set_num_threads(1)
    r=a.runtime;out=r/'results';out.mkdir(exist_ok=True)
    ckpath=r/'checkpoints/PAPER_QB_s42/best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    history=json.loads((r/'history.json').read_text(encoding='utf-8'))
    minimum=min(x['val_mse'] for x in history);assert ck['validation_mse']==minimum and ck['epoch']==next(x['epoch'] for x in history if x['val_mse']==minimum)
    net=PaperSSAMRN().cuda().eval();net.load_state_dict(ck['model'],strict=True)
    write(out/'selection.json',{'rule':'validation minimum; earliest exact tie','epoch':ck['epoch'],'checkpoint_sha256':sha(ckpath),'example_indices_zero_based':[1,8,11,13,19],'no_test_tuning':True})
    summaries={}
    for protocol,filename in [('RR','test_qb_multiExm1.h5'),('FR','test_qb_OrigScale_multiExm1.h5')]:
        paths=list(a.data.rglob(filename));assert len(paths)==1;rows=[]
        with h5py.File(paths[0],'r') as f,torch.inference_mode():
            assert len(f['pan'])==20
            for i in range(20):
                keys=('pan','lms','ms','gt') if protocol=='RR' else ('pan','lms','ms')
                arr={k:np.asarray(f[k][i],dtype=np.float32) for k in keys}
                p,l,m=[torch.from_numpy(arr[k]/2047.).unsqueeze(0).cuda() for k in ('pan','lms','ms')]
                pred=net(p,l,m)[0].cpu().numpy().astype(np.float64)*2047.;assert np.isfinite(pred).all()
                # Fixed H5 LMS in FR metrics preserves the historical observation reference.
                metrics=rr_metrics(arr['gt'].astype(np.float64),pred) if protocol=='RR' else fr_metrics(pred,arr['lms'],arr['ms'],arr['pan'])
                assert all(np.isfinite(x) for x in metrics.values());rows.append({'index':i,'metrics':metrics})
                if protocol=='RR' and i in (1,8,11,13,19):panels(arr,pred,out/f'RR_scene_{i:03}.png','Paper equation reproduction',i)
                print(protocol,i,metrics,flush=True)
        report={'samples':20,'data_sha256':sha(paths[0]),'rows':rows,'mean':{k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']},'precision':'FP32 forward; no clipping','FR_reference':'original H5 LMS fixed, not candidate-dependent','limits':'MATLAB parity unverified; historical exploratory test'}
        write(out/f'{protocol}.json',report);summaries[protocol]=report['mean']
    write(out/'summary.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4));epochs=[x['epoch'] for x in history]
    axes[0].plot(epochs,[x['train_mse'] for x in history],label='train');axes[0].plot(epochs,[x['val_mse'] for x in history],label='validation');axes[0].legend();axes[0].set_yscale('log');axes[0].set_title('MSE')
    for ax,key in zip(axes[1:],('val_psnr','val_sam')):ax.plot(epochs,[x[key] for x in history]);ax.set_title(key)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    write(r/'complete.json',{'training_epochs':100,'evaluation_samples':{'RR':20,'FR':20},'best_epoch':ck['epoch'],'best_sha256':sha(ckpath),'metrics':summaries})
