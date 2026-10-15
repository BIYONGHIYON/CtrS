"""Verify retained FFL evidence without training or original H5 data."""
import sys,json,hashlib,math
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=read(ROOT/'manifest.json')
    for n,e in m['files'].items():
        p=ROOT/n
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('Artifact mismatch: '+n)
    import torch,numpy as np
    torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'provenance/code/SSA-MRN/src'))
    from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
    safe=[np.ndarray,np.dtype,np._core.multiarray._reconstruct,np.dtypes.UInt32DType]
    count=0
    with torch.serialization.safe_globals(safe):
        for p in sorted((ROOT/'checkpoints').rglob('*.pt')):
            ck=torch.load(p,map_location='cpu',weights_only=True);state=ck['model'] if 'model' in ck else ck
            net=RestoredPansharpeningNet(channels=4,ssai_dimension=4);net.load_state_dict(state,strict=True);net.eval()
            with torch.inference_mode():y=net(torch.zeros(1,1,64,64),torch.zeros(1,4,64,64),torch.zeros(1,4,16,16))
            assert y.shape==(1,4,64,64) and torch.isfinite(y).all();count+=1
    assert count==14
    for arm in ('baseline','ffl'):
        h=read(ROOT/f'results/final/{arm}/history.json');assert len(h)==100
        r=read(ROOT/f'results/final/{arm}/results.json')
        assert r['best_val_mse']==min(x['val_mse'] for x in h)
        for split in ('RR','FR'):
            rows=read(ROOT/f'results/final/{arm}/{split}_per_scene.json');assert len(rows)==20
            for k,v in r['splits'][split]['mean'].items():assert math.isclose(v,float(np.mean([x[k] for x in rows])),abs_tol=1e-12)
    seal=read(ROOT/'provenance/final_settings.json');mapping=read(ROOT/'results/source_artifact_map.json')
    for original,expected in seal['evidence_sha256'].items():assert sha(ROOT/mapping[original])==expected
    assert not seal['test_used'] and seal['chosen_lambda']==0.013491834
    print('PASS: retained hashes, 14 strict loads/CPU forwards, validation selection and 20-scene means; no original H5 re-evaluation')
if __name__=='__main__':main()
