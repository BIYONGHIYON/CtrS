import sys,json,numpy as np,torch
X='C:/CtrS-p69/SSA-MRN/experiments/performance/69_low_frequency_squared_error_loss'
sys.path.insert(0,X)
import train
cfg=json.load(open('C:/CtrS-p69/run/runtime_config.json',encoding='utf8'))
T=train.T
for sensor in ('QB','WV3'):
    tr=T.cache(cfg,sensor,'train');trc=T.cache(cfg,sensor,'train',True);idx=np.arange(0,len(tr['pan']),997)[:16]
    b=[torch.from_numpy(np.array(tr[k][idx])).cuda() for k in T.KEYS];bc=[torch.from_numpy(np.array(trc[k][idx])).cuda() for k in T.KEYS]
    PH=train.phase_table(cfg,sensor,len(tr['pan']))[idx[:2]]
    for arm in T.ARMS:
        T.setup(42);m=T.make_model(arm,sensor,42,cfg['layout']);m.train()
        inp=[x[:2].contiguous(memory_format=torch.channels_last) for x in b[:3]];ci=[x[:2].contiguous(memory_format=torch.channels_last) for x in bc[:3]]
        for scale in (1024.,64.,1.):
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):p=m(*inp);pc=m(*ci)
            parts=dict(rec=train.recon(cfg,arm,sensor,p.float()-b[3][:2],PH).mean(),coarse=train.recon(cfg,arm,sensor,pc.float()-bc[3][:2]).mean(),spec=T.spectral_loss(p.float(),b[3][:2]))
            loss=parts['rec']+parts['coarse']+0.46*parts['spec']
            (loss*scale).backward()
            g=[x.grad for x in m.parameters() if x.grad is not None]
            fin=all(torch.isfinite(x).all() for x in g);mx=max(float(x.abs().max()) for x in g if torch.isfinite(x).all())
            print(sensor,arm,scale,{k:float(v) for k,v in parts.items()},'p finite',bool(torch.isfinite(p).all()),'grads finite',fin,'max finite grad/scale',mx/scale,flush=True)
