"""Verify retained evidence and synthetic CPU outputs; no training or private-H5 reevaluation."""
import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=read(ROOT/'manifest.json')
    for n,v in m['files'].items():
        p=ROOT/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
    orig=read(ROOT/'provenance/original_artifact_manifest.json')
    for n,target in read(ROOT/'results/source_artifact_map.json').items():
        if n in orig:assert sha(ROOT/target)==orig[n]['sha256'],n
    import torch,numpy as np
    torch.set_num_threads(1);sys.path.insert(0,str(ROOT))
    from model import Model
    from hf_gate_worker import state_hash
    safe=[np.ndarray,np.dtype,np._core.multiarray._reconstruct,np.dtypes.UInt32DType]
    count=0
    with torch.serialization.safe_globals(safe):
        for seed in (42,43,44):
            ini=torch.load(ROOT/f'checkpoints/initial_QB_seed{seed}/initial.pt',map_location='cpu',weights_only=True)
            model=Model('baseline');model.load_state_dict(ini,strict=True);count+=1;orders=[]
            for arm in ('baseline','hf_gate'):
                base=ROOT/f'results/seed_{seed}/{arm}';history=read(base/'history.json')
                init=read(base/'initialization.json');assert init['common_state_sha256']==state_hash(ini) and init['parameters']==2853951
                assert [r['epoch'] for r in history]==list(range(101))
                proof=read(base/'evaluation_complete.json');selected=min(history,key=lambda h:h['val_mse'])
                for kind in ('best','latest'):
                    p=ROOT/f'checkpoints/{arm}_QB_seed{seed}/{kind}.pt';assert sha(p)==proof[kind+'_sha256']
                    ck=torch.load(p,map_location='cpu',weights_only=True)
                    model=Model(arm,seed=seed);assert model.core.backbone.arm==arm;model.load_state_dict(ck['model'],strict=True);model.eval()
                    with torch.inference_mode():y=model(torch.zeros(1,1,64,64),torch.zeros(1,4,64,64),torch.zeros(1,4,16,16))
                    assert y.shape==(1,4,64,64) and torch.isfinite(y).all();count+=1
                    if arm=='baseline':assert all(torch.equal(v,ini[k]) for k,v in ck['model'].items() if '.gate.' in k)
                    else:assert any(not torch.equal(v,ini[k]) for k,v in ck['model'].items() if '.gate.' in k)
                    if kind=='best':assert ck['epoch']==selected['epoch'] and ck['mse']==selected['val_mse']
                    else:assert ck['epoch']==100 and ck['next_sample_position']==0;orders.append(ck['order_history'])
                    eb=base/'evaluations'/kind;r=read(eb/'results.json');assert sha(eb/'results.json')==proof['results_sha256'][kind]
                    assert not r['smoke_validation_inputs_only'] and r['best_epoch']==ck['epoch']
                    for split in ('RR','RR2','FR_WALD','FR'):
                        rows=read(eb/(split+'_per_scene.json'));assert [x['scene_index'] for x in rows]==list(range(20))
                        for k,v in r['splits'][split]['mean'].items():assert math.isclose(v,float(np.mean([x[k] for x in rows])),abs_tol=1e-12)
            assert orders[0]==orders[1]
    from evaluate import calculate
    assert calculate(ROOT)==read(ROOT/'results/judgement_v2_3.json')
    assert count==15
    print('PASS: retained hashes, 15 strict loads/12 CPU forwards, 3-seed validation best, paired orders, all4 tests20-scene means, gate learning and v2.3 recomputation; no original H5 reevaluation')
if __name__=='__main__':main()
