"""Validate retained hashes, selected epochs, actual batch orders and test aggregation."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,e in manifest['files'].items():
        b=(ROOT/name).read_bytes()
        assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
    hist={a:json.loads((ROOT/'results'/a/'history.json').read_text()) for a in ('P0','P1')}
    assert [r['order_sha256'] for r in hist['P0'][1:]]==[r['order_sha256'] for r in hist['P1'][1:]]
    for a in ('P0','P1'):
        h=hist[a];assert [r['epoch'] for r in h]==list(range(21))
        best=min(h,key=lambda r:r['val_mse'])
        marker=json.loads((ROOT/'results'/a/'complete.json').read_text())
        assert marker['best_epoch']==best['epoch']
        for name in ('best','latest'):
            assert hashlib.sha256((ROOT/'checkpoints'/f'{a}_QB_s46'/f'{name}.pt').read_bytes()).hexdigest()==marker[name+'_sha256']
        assert h[-1]['updates']==h[-1]['attempted']==5360 and h[-1]['skipped']==0
        metrics=json.loads((ROOT/'results'/a/'results/metrics.json').read_text())
        for proto in ('RR','FR'):
            r=metrics['protocols'][proto];assert len(r['rows'])==20
            for k,v in r['mean'].items():assert math.isclose(sum(x['metrics'][k] for x in r['rows'])/20,v,rel_tol=1e-12,abs_tol=1e-12)
        for i in (1,8,11,13,19):assert (ROOT/'results'/a/'results'/f'RR_scene_{i:03}.png').is_file()
    print(f"PASS: {len(manifest['files'])} file hashes, 5 checkpoints, paired orders and complete RR/FR records; no fresh model inference")
if __name__=='__main__':main()
