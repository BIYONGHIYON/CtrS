"""Verify preserved originals; regenerate summaries/figures from recorded results."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
from model import Model,ARMS
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    return a==b

def verify():
    torch.set_num_threads(1);export=read(HERE/'provenance/server_export_manifest.json')
    for name,e in export.items():
        p=HERE/name
        assert p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],name
    # Ensure the readable implementation equals the source that really ran.
    for name in ('model.py','train.py','run.py','config.json'):
        assert sha(HERE/name)==sha(HERE/'provenance/training_source'/name),name
    cfg=read(HERE/'provenance/runtime_config.json');summary={};deltas={};orders={};initial_core={};preds={}
    torch.manual_seed(884);inputs=[torch.rand(s) for s in [(1,1,64,64),(1,4,64,64),(1,4,16,16)]]
    checks={}
    for arm in ARMS:
        tag=f'{arm}_QB_s42';r=HERE/'results'/tag;cp=HERE/'checkpoints'/tag
        h=read(r/'history.json');assert [x['epoch'] for x in h]==list(range(101))
        selected=min(h,key=lambda x:x['val_mse']);done=read(r/'training_complete.json');edone=read(r/'evaluation_complete.json')
        assert done['epoch']==100 and done['attempted']==done['updates']==53600
        assert done['fingerprint']==cfg['fingerprint']
        assert done['best_epoch']==edone['best_epoch']==selected['epoch']
        assert done['best_sha256']==edone['weight_sha256']==sha(cp/'best.pt')
        assert done['latest_sha256']==sha(cp/'latest.pt')
        best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
        latest=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['epoch']==100 and latest['next_batch']==0 and latest['attempted']==latest['updates']==53600
        assert latest['best_epoch']==selected['epoch'] and same(latest['best_model'],best['model'])
        assert all(float(st['step'])==53600 for st in latest['optimizer']['state'].values())
        orders[arm]=[x['order_sha256'] for x in h[1:]]
        m=Model(arm,42).eval();counts={}
        for name in ('initial','best','latest'):
            ck=torch.load(cp/f'{name}.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True)
            with torch.inference_mode():y=m(*inputs)
            assert y.shape==inputs[1].shape and torch.isfinite(y).all()
            counts[name]=True
            if name=='initial':
                initial_core[arm]={k:v for k,v in ck['model'].items() if '.branches.' not in k and k!='core.gates'}
                preds[arm]=y
        means={};protocols={}
        for proto in ('RR','FR'):
            data=read(r/'evaluation'/f'{proto}.json');assert data['weight_sha256']==sha(cp/'best.pt') and data['best_epoch']==selected['epoch']
            assert [x['index'] for x in data['rows']]==list(range(20))
            for k,v in data['mean'].items():assert math.isclose(np.mean([x['metrics'][k] for x in data['rows']]),v,rel_tol=0,abs_tol=1e-12)
            assert data['mean']==edone['means'][proto]
            means[proto]=data['mean'];protocols[proto]=data
        for i in cfg['scenes']:assert (r/'evaluation'/f'RR_scene_{i:03}.png').is_file()
        summary[arm]={'best_epoch':selected['epoch'],'validation':{k:selected[k] for k in ('val_mse','val_psnr','val_sam')},'test':means,'best_sha256':sha(cp/'best.pt'),'parameters':sum(p.numel() for p in m.parameters()),'updates':53600,'skips':0,'train_seconds_sum':sum(x['epoch_seconds'] for x in h[1:])}
        checks[arm]={'strict_cpu_finite_checkpoints':counts,'epoch_selection_verified':True,'optimizer_updates_verified':53600,'full_RR_FR_scene_counts':[20,20]}
    for arm in ARMS:
        assert orders[arm]==orders['B0'] and same(initial_core[arm],initial_core['B0'])
        torch.testing.assert_close(preds[arm],preds['B0'],rtol=0,atol=0)
    assert len({summary[a]['parameters'] for a in ('C0','F1','F2')})==1
    for a,b in [('C0','B0'),('F1','B0'),('F2','B0'),('F2','C0'),('F2','F1')]:
        paired={}
        for proto in ('RR','FR'):
            aa=read(HERE/'results'/f'{a}_QB_s42/evaluation/{proto}.json');bb=read(HERE/'results'/f'{b}_QB_s42/evaluation/{proto}.json')
            rows=[{'index':x['index'],'delta':{k:x['metrics'][k]-y['metrics'][k] for k in x['metrics']}} for x,y in zip(aa['rows'],bb['rows'])]
            mean={k:float(np.mean([x['delta'][k] for x in rows])) for k in aa['mean']}
            wins={k:sum(x['delta'][k]>0 if k in ('PSNR','SCC','Q2n','QNR') else x['delta'][k]<0 for x in rows) for k in mean}
            paired[proto]={'mean_delta':mean,'improved_scenes':wins,'rows':rows}
        deltas[f'{a}_vs_{b}']=paired
    checks={'export_files_verified':len(export),'original_checkpoints':12,'common_core_initial_equal':True,'initial_CPU_outputs_equal':True,'all_epoch_orders_equal':True,'checks':checks,'scope':'hashes, recorded metric means, selection, strict load and finite synthetic CPU inference; no new H5 inference or MATLAB parity verification'}
    return summary,deltas,checks

def figures(summary,deltas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    out=HERE/'results/figures';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    records={a:read(HERE/'results'/f'{a}_QB_s42/history.json') for a in ARMS}
    colors={'B0':'#6b7280','C0':'#d49a32','F1':'#2c8999','F2':'#845ec2'}
    for a,h in records.items():
        epochs=[x['epoch'] for x in h[1:]]
        for ax,k in zip(axes[0],('train_mse','val_psnr','val_sam')):ax.plot(epochs,[x[k] for x in h[1:]],label=a,color=colors[a],lw=1.2)
        if a!='B0':
            for ax,k in zip(axes[1],('train_mse','val_psnr','val_sam')):ax.plot(epochs,[x[k]-b[k] for x,b in zip(h[1:],records['B0'][1:])],label=a,color=colors[a],lw=1.1)
    for ax,title in zip(axes[0],('Training MSE','Validation PSNR (dB)','Validation SAM (deg)')):ax.set(title=title,xlabel='Epoch');ax.legend()
    for ax,title in zip(axes[1],('Training MSE: candidate minus B0','Validation PSNR: candidate minus B0','Validation SAM: candidate minus B0')):
        ax.axhline(0,color='black',lw=.8);ax.set(title=title,xlabel='Epoch');ax.legend()
    axes[0,0].set_yscale('log');fig.suptitle('QB seed42 / FP32 / identical sample order / epochs 1-100');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    metrics=[('RR','PSNR','up'),('RR','SAM','down'),('RR','ERGAS','down'),('RR','Q2n','up'),('RR','SCC','up'),('FR','QNR','up'),('FR','D_lambda','down'),('FR','D_s','down')]
    for ax,(proto,k,direction) in zip(axes.flat,metrics):
        vals=[summary[a]['test'][proto][k] for a in ARMS];bars=ax.bar(ARMS,vals,color=['#6b7280','#d49a32','#2c8999','#845ec2']);ax.set_title(f'{proto} {k} ({direction})');ax.set_ylim(min(vals)-max((max(vals)-min(vals))*.6,abs(min(vals))*.001),max(vals)+max((max(vals)-min(vals))*.6,abs(max(vals))*.001))
        for bar,v in zip(bars,vals):ax.annotate(f'{v:.5f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('Full 20-scene means / FR metrics provisional / axes zoomed');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(proto,k) in zip(axes,[('RR','PSNR'),('RR','SAM'),('FR','QNR')]):
        for a in ARMS[1:]:ax.plot(range(20),[x['delta'][k] for x in deltas[f'{a}_vs_B0'][proto]['rows']],marker='.',label=a,color=colors[a])
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'{proto} {k}: candidate minus B0',xlabel='Scene index (0-based)');ax.legend()
    fig.tight_layout();fig.savefig(out/'scene_deltas.png',dpi=160);plt.close(fig)

def manifest():
    files={}
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts:
            files[p.relative_to(HERE).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    write(HERE/'manifest.json',{'experiment':44,'research_track':'performance','files':files})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    summary,deltas,checks=verify()
    if not args.verify_only:
        write(HERE/'results/summary.json',summary);write(HERE/'results/paired_scene_deltas.json',deltas);write(HERE/'results/verification.json',checks);figures(summary,deltas);manifest()
    print(json.dumps(checks,indent=2))
