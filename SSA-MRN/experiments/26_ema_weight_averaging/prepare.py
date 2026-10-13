"""Offline retained evidence verification; never starts training or overwrites results."""
import hashlib,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=read(ROOT/'manifest.json')
    for name,e in manifest['files'].items():
        p=ROOT/name
        if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:raise ValueError('File mismatch: '+name)
    import numpy as np,torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'provenance/code/SSA-MRN/src'))
    from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
    def statehash(s):
        h=hashlib.sha256()
        for k,v in sorted(s.items()):
            h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.contiguous().numpy().tobytes())
        return h.hexdigest()
    safe=[np.ndarray,np.dtype,np._core.multiarray._reconstruct,np.dtypes.UInt32DType]
    weights={}
    with torch.serialization.safe_globals(safe):
        for arm in ('baseline','ema'):
            weights[arm]={}
            for kind in ('best','latest'):
                ck=torch.load(ROOT/f'checkpoints/{arm}_QB_s42/{kind}.pt',map_location='cpu',weights_only=True)
                weights[arm][kind]=ck
                net=RestoredPansharpeningNet(channels=4,ssai_dimension=4);net.load_state_dict(ck['model'],strict=True);net.eval()
                with torch.inference_mode():y=net(torch.zeros(1,1,64,64),torch.zeros(1,4,64,64),torch.zeros(1,4,16,16))
                if y.shape!=(1,4,64,64) or not torch.isfinite(y).all():raise ValueError('Invalid synthetic output')
            h=read(ROOT/f'results/{arm}/history.json')
            assert len(h)==100 and [r['epoch'] for r in h]==list(range(1,101))
            assert weights[arm]['best']['mse']==min(r['val_mse'] for r in h)
            result=read(ROOT/f'results/{arm}/results.json')
            for split in ('RR','FR'):
                rows=read(ROOT/f'results/{arm}/{split}_per_scene.json')
                assert len(rows)==20 and [r['scene_index'] for r in rows]==list(range(20))
                for k,v in result['splits'][split]['mean'].items():assert math.isclose(v,float(np.mean([r[k] for r in rows])),abs_tol=1e-12)
    assert statehash(weights['baseline']['latest']['model'])==statehash(weights['ema']['latest']['model'])
    assert weights['baseline']['latest']['order_history']==weights['ema']['latest']['order_history']
    assert statehash(weights['ema']['best']['model'])==statehash(weights['ema']['latest']['ema_state']['shadows']['0.99'])
    print('PASS: retained hashes, 4 strict loads/synthetic CPU forwards, 100 epochs, online/order equivalence, EMA best, 20-scene means; no original H5 reevaluation')
if __name__=='__main__':main()
