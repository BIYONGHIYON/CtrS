"""Initial-function parity, gate gradients, AMP/Adam recovery and batched mmap IO."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
CoarseAnchorNet = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.model').CoarseAnchorNet; VARIANTS = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.model').VARIANTS
checkpoint = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run').checkpoint; seed = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run').seed; Cached = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run').Cached; Remaining = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run').Remaining; loader = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run').loader

def main():
    seed();torch.backends.cudnn.benchmark=False
    pan=torch.rand(2,1,64,64,device='cuda');lms=torch.rand(2,4,64,64,device='cuda');ms=torch.rand(2,4,16,16,device='cuda')
    models={}
    for v in VARIANTS:
        torch.manual_seed(42);models[v]=CoarseAnchorNet(v).cuda().eval()
    for a,b in [('N3','N2'),('N4','N0'),('N5','N2')]:
        with torch.no_grad():torch.testing.assert_close(models[a](pan,lms,ms),models[b](pan,lms,ms),rtol=0,atol=0)
    rows=[]
    for v,m in models.items():
        y=m(pan,lms,ms);assert y.shape==lms.shape and torch.isfinite(y).all()
        y.square().mean().backward()
        for k in ('coarse_gain','lrpan_gain'):
            if hasattr(m,k):
                g=getattr(m,k).grad;assert g is not None and torch.isfinite(g) and g.abs()>0,(v,k,g)
        rows.append(dict(variant=v,parameters=sum(p.numel() for p in m.parameters()),gates=m.gate_values()))
    m=models['N5'].train();o=torch.optim.Adam(m.parameters(),lr=1e-4);sc=torch.amp.GradScaler('cuda')
    def step(m,o,sc):
        o.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): y=m(pan,lms,ms)
        sc.scale(y.float().square().mean()).backward();sc.step(o);sc.update()
    step(m,o,sc)
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'latest.pt';checkpoint(f,dict(model=m.state_dict(),optimizer=o.state_dict(),scaler=sc.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        step(m,o,sc);expected={k:v.clone() for k,v in m.state_dict().items()}
        ck=torch.load(f,map_location='cpu',weights_only=False);r=CoarseAnchorNet('N5').cuda();r.load_state_dict(ck['model']);ro=torch.optim.Adam(r.parameters(),lr=1e-4);ro.load_state_dict(ck['optimizer']);rs=torch.amp.GradScaler('cuda');rs.load_state_dict(ck['scaler']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);step(r,ro,rs)
        resume_max_abs=max(float((x-expected[k]).abs().max()) for k,x in r.state_dict().items())
        for k,x in r.state_dict().items():torch.testing.assert_close(x,expected[k],rtol=1e-6,atol=1e-8)
        assert sc.state_dict()==rs.state_dict()
        assert o.state_dict()["param_groups"]==ro.state_dict()["param_groups"]
        folder=Path(d)/'mmap';folder.mkdir()
        for k in ('pan','lms','ms','gt'):np.save(folder/(k+'.npy'),np.arange(17,dtype=np.float32).reshape(17,1,1,1))
        ds=Cached(folder);sampler=Remaining(17,0,0,4);dl=loader(ds,2,sampler)
        first=torch.cat([b[0].flatten() for b in dl]).tolist()
        sampler.epoch=1;sampler.start=2
        second=torch.cat([b[0].flatten() for b in dl]).tolist()
        assert first==torch.randperm(17,generator=torch.Generator().manual_seed(42)).tolist()
        assert second==torch.randperm(17,generator=torch.Generator().manual_seed(43)).tolist()[8:]
        del dl
    from types import SimpleNamespace
    from unittest.mock import patch
    suite = __import__('importlib').import_module('experiments.18_coarse_feature_fusion.run')
    decisions=[]
    for eligible in (False,True):
        with tempfile.TemporaryDirectory() as directory:
            rt=Path(directory); launched=[]
            values={'N0':1.,'N1':.95,'N2':.9,'N3':.8 if eligible else .91,'N4':.85,'N5':.7}
            class FakeProcess:
                def __init__(self,args,**kwargs):
                    v=args[args.index('--variant')+1];launched.append(v);self.pid=99999;self.returncode=0
                    suite.write(rt/v/'complete.json',dict(best_mse=values[v]))
                def poll(self):return 0
            with patch.object(suite.subprocess,'Popen',FakeProcess),patch.object(suite.time,'sleep',lambda _:None),patch.object(suite,'evaluate',lambda a:None):
                suite.controller(SimpleNamespace(runtime=rt))
            assert ('N5' in launched)==eligible
            assert all(v in launched for v in VARIANTS[:5])
            decisions.append(dict(eligible=eligible,launched=launched))
    result=dict(initial_parity=True,gate_gradients=True,amp_adam_resume_close=True,resume_rtol=1e-6,resume_atol=1e-8,resume_max_abs=resume_max_abs,persistent_worker_epoch_resume_order=True,rows=rows,conditional_queue=decisions)
    print(json.dumps(result),flush=True)
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
