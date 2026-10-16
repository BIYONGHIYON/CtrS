"""Verify retained artifacts and synthetic inference only; no training or original-H5 reevaluation."""
import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=read(ROOT/'manifest.json')
    for n,v in manifest['files'].items():
        p=ROOT/n;assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],n
    mapping=read(ROOT/'results/source_artifact_map.json')
    original=read(ROOT/'provenance/original_artifact_manifest.json')
    for n,rel in mapping.items():
        if n in original:assert digest(ROOT/rel)==original[n]['sha256'],rel
    import torch,numpy as np
    torch.set_num_threads(1);sys.path.insert(0,str(ROOT))
    from model import W2Net,common_state
    safe=[np.ndarray,np.dtype,np._core.multiarray._reconstruct,np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(safe):
        initial=torch.load(ROOT/'checkpoints/initial_QB_seed42/initial.pt',map_location='cpu',weights_only=True)
        net=W2Net(False);net.load_state_dict(initial,strict=True);net.eval()
        with torch.inference_mode():assert torch.isfinite(net(torch.zeros(1,1,64,64),torch.zeros(1,4,64,64),torch.zeros(1,4,16,16))).all()
        orders=[]
        for arm in ('baseline','gated'):
            history=read(ROOT/f'results/{arm}/history.json');assert len(history)==150
            proof=read(ROOT/f'results/{arm}/evaluation_complete.json')
            for kind in ('best','latest'):
                cp=ROOT/f'checkpoints/{arm}_QB_seed42/{kind}.pt'
                assert digest(cp)==proof[kind+'_sha256']
                ck=torch.load(cp,map_location='cpu',weights_only=True)
                net=W2Net(arm=='gated');net.load_state_dict(ck['model'],strict=True);net.eval()
                with torch.inference_mode():
                    y=net(torch.zeros(1,1,64,64),torch.zeros(1,4,64,64),torch.zeros(1,4,16,16))
                assert y.shape==(1,4,64,64) and torch.isfinite(y).all()
                if kind=='best':assert ck['mse']==min(x['val_mse'] for x in history)
                else:assert ck['epoch']==150;orders.append(ck['order_history'])
                p=ROOT/f'results/{arm}/evaluations/{kind}'
                assert digest(p/'results.json')==proof['results_sha256'][kind]
                r=read(p/'results.json');assert not r['smoke_validation_inputs_only']
                for split in ('RR','FR'):
                    rows=read(p/(split+'_per_scene.json'));assert [x['scene_index'] for x in rows]==list(range(20))
                    for k,v in r['splits'][split]['mean'].items():assert math.isclose(v,float(np.mean([x[k] for x in rows])),abs_tol=1e-12)
            for i in (1,8,11,13,19):assert (ROOT/f'results/{arm}/evaluations/best/RR_scene_{i:03}.png').is_file()
        assert orders[0]==orders[1]
    print('PASS: retained hashes, 5 strict loads and CPU forwards, validation selection, matched orders, 20-scene means. Original H5 not re-evaluated.')
if __name__=='__main__':main()
