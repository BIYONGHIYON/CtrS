"""Verify original checkpoint hashes, validation selection and complete QB test evidence."""
import argparse, hashlib, json, math
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1]/'docs/assets/windows_losses')
    a=p.parse_args(); root=a.assets
    manifest=json.loads((root/'report_manifest.json').read_text())
    for name,item in manifest['artifacts'].items():
        file=root/name.replace('\\','/'); assert file.is_file(),name
        assert file.stat().st_size==item['bytes'] and sha(file)==item['sha256'],name
    assert sha(root/'server_inventory.json')==manifest['server_inventory_sha256']
    original=json.loads((root/'server_inventory.json').read_text())
    assert original['artifacts']=={**manifest['artifacts'],**manifest['excluded_runtime_caches']}
    assert all('__pycache__' in n and n.endswith('.pyc') for n in manifest['excluded_runtime_caches'])
    report=json.loads((root/'metrics.json').read_text())
    assert len(report['runs'])==10 and manifest['original_checkpoint_count']==20
    final=json.loads((root/'final_combined_history.json').read_text())
    for name,r in report['runs'].items():
        d=root/'runs'/name; c=json.loads((d/'config.json').read_text()); complete=json.loads((d/'complete.json').read_text())
        assert r['config']==c==complete['config'] and c['k']==4 and c['seed']==42
        rows=final if name=='05_final_candidate' else [json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in rows]==list(range(1,c['epochs']+1))
        assert all(math.isfinite(x[k]) for x in rows for k in ['train_mse','val_mse','seconds'])
        best=min(rows,key=lambda x:x['val_mse']); assert r['best_epoch']==best['epoch']
        assert complete['best_mse']==r['best_validation_mse']==best['val_mse']
        for kind,item in r['checkpoints'].items():
            assert item['epoch']==(best['epoch'] if kind=='best' else c['epochs'])
            assert sha(d/f'{kind}.pt')==item['sha256']
    screened={n:r for n,r in report['runs'].items() if n!='05_final_candidate'}
    winner=min(screened,key=lambda n:(screened[n]['best_validation_mse'],n))
    selection=json.loads((root/'screening_selection.json').read_text())
    assert winner==selection['winner']['name']=='04_edge_w0.1_s42'
    continuation=json.loads((root/'runs/05_final_candidate/continuation.json').read_text())
    assert continuation=={'source':winner,'resume':'latest','start_epoch':30}
    base=json.loads((root/'baseline_reference.json').read_text())
    assert sha(root.parent/'windows_k_baselines/runs/01_QB_k4_s42/best.pt')==base['checkpoint_sha256']==report['test']['baseline']['checkpoint_sha256']
    scenes=json.loads((root/'scene_selection.json').read_text())['zero_based_scene_indices']
    assert len(scenes)==len(set(scenes))==5
    for name,r in report['test'].items():
        assert [x['index'] for x in r['images']]==scenes
        for protocol,record in r['metrics'].items():
            assert record['count']==report['data'][protocol]['shapes']['pan'][0]==len(record['rows'])==20
            assert [x['index'] for x in record['rows']]==list(range(20))
            for key,mean in record['mean'].items():
                values=[x['metrics'][key] for x in record['rows']]; assert all(math.isfinite(x) for x in values)
                assert math.isclose(sum(values)/len(values),mean,rel_tol=1e-12,abs_tol=1e-14)
                if name=='edge_w0.1': assert math.isclose(mean-report['test']['baseline']['metrics'][protocol]['mean'][key],report['delta_from_baseline'][protocol][key],rel_tol=1e-12,abs_tol=1e-14)
    assert json.loads((root/'evaluation_state.json').read_text())['status']=='finished'
    print(f'PASS: {len(manifest["artifacts"])} original files, 20 full checkpoints, 940 planned training epochs, 80 full-scene evaluations, 10 four-panel images')
if __name__=='__main__':main()
