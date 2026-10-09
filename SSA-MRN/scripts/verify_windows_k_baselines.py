"""Verify the preserved six Windows baselines without loading CUDA or running training."""
from pathlib import Path
import argparse,hashlib,json,math

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1]/'docs/assets/windows_k_baselines')
    args=parser.parse_args();root=args.assets;manifest=json.loads((root/'manifest.json').read_text())
    for name,item in manifest['files'].items():
        p=root/name
        assert p.is_file(),name
        assert p.stat().st_size==item['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
    assert len(manifest['runs'])==6
    wins={4:0,6:0}
    for name,item in manifest['runs'].items():
        d=root/'runs'/name;config=json.loads((d/'config.json').read_text());complete=json.loads((d/'complete.json').read_text())
        rows=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        assert config==item['config']==complete['config']
        assert [x['epoch'] for x in rows]==list(range(1,101))
        assert all(math.isfinite(x[k]) for x in rows for k in ['train_mse','val_mse','seconds'])
        best=min(rows,key=lambda x:x['val_mse'])
        assert complete['epoch']==item['epochs']==100
        assert best['epoch']==item['best_epoch']
        assert best['val_mse']==complete['best_mse']==item['best_validation_mse']
        for kind in ['best','latest']:
            ck=item['checkpoints'][kind]
            assert ck['epoch']==(best['epoch'] if kind=='best' else 100)
            assert ck['sha256']==manifest['files'][f'runs/{name}/{kind}.pt']['sha256']
    selection=json.loads((root/'baseline_selection.json').read_text())
    for sensor in ['QB','GF2','WV3']:
        scores={k:manifest['runs'][f'01_{sensor}_k{k}_s42']['best_validation_mse'] for k in [4,6]}
        for k,value in scores.items():assert selection['scores'][f'{sensor}_k{k}']==value
        wins[4 if scores[4]<=scores[6] else 6]+=1
    assert selection['k']==(4 if wins[4]>=wins[6] else 6)==4
    print(f'PASS: {len(manifest["files"])} source/data files, 12 checkpoint hashes, 600 epochs; K4 wins {wins[4]}:{wins[6]}')

if __name__=='__main__':main()
