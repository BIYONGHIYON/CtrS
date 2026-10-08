"""Verify report completeness, paired arithmetic, full checkpoints and training code provenance."""
import argparse, hashlib, json, math
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args()
    out=a.root/'docs/assets/a6000_gf2_repeat_qb_input'; m=json.loads((out/'metrics.json').read_text()); manifest=json.loads((out/'report_manifest.json').read_text())
    assert m['completed_new_runs'] in (4,6) and len(m['runs'])==m['completed_new_runs']+9
    assert sum(not r['reused'] for r in m['runs'].values())==m['completed_new_runs']
    assert manifest['training_complete'] and manifest['evaluation_complete']
    for name,artifact in manifest['artifacts'].items():
        path=out/name; assert path.stat().st_size==artifact['bytes'] and digest(path)==artifact['sha256'],name
    assert set(manifest['artifacts'])=={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file() and f.name!='report_manifest.json'}
    for name,sha in manifest['code'].items(): assert digest(a.root/name)==sha,name
    inventory=json.loads((out/'training_inventory.json').read_text())
    for name,sha in inventory['source'].items(): assert digest(a.root/name)==sha,name
    weight_count=0; image_count=0
    for run_id,r in m['runs'].items():
        assert [x['epoch'] for x in r['history']]==list(range(1,101)),run_id
        b=r['checkpoints']['best']; assert b['epoch']==min(r['history'],key=lambda x:x['val_mse'])['epoch']
        assert math.isclose(b['best_mse'],min(x['val_mse'] for x in r['history']),rel_tol=1e-12)
        assert r['checkpoints']['latest']['epoch']==100
        if r['config']['sensor']=='QB' and r['config']['seed']==42 and r['config']['variant'] in ['baseline','interp23']: assert all('val_sam_deg' not in h and 'val_psnr_band_mean_peak1' not in h for h in r['history'])
        else: assert all(math.isfinite(h[k]) for h in r['history'] for k in ['train_mse','val_mse','val_sam_deg','val_psnr_band_mean_peak1'])
        for ck in r['checkpoints'].values(): assert digest(out/ck['file'])==ck['sha256']; weight_count+=1
        for protocol,metrics in r['metrics'].items():
            count=m['data'][r['config']['sensor']][protocol]['shapes']['pan'][0]
            assert metrics['count']==count==len(metrics['rows'])
            assert [x['index'] for x in metrics['rows']]==list(range(count))
            baseline=m['runs'][f'{r["config"]["sensor"]}_baseline_k6_s{r["config"]["seed"]}']['metrics'][protocol]['mean']
            for key,value in metrics['mean'].items():
                assert math.isfinite(value)
                assert math.isclose(value,sum(x['metrics'][key] for x in metrics['rows'])/count,rel_tol=1e-12,abs_tol=1e-12)
                assert math.isclose(r['delta_from_baseline'][protocol][key],value-baseline[key],rel_tol=1e-12,abs_tol=1e-12)
        if r['config']['variant']!='baseline':
            assert [im['index'] for im in r['images']]==m['selection'][r['config']['sensor']]['zero_based_scene_indices']
            assert len(r['images'])==5; image_count+=5
    for group,protocols in m['paired_summaries'].items():
        for protocol,keys in protocols.items():
            for key,s in keys.items():
                count=len(s['values']);assert count==len(s['seeds'])
                assert math.isclose(s['mean'],sum(s['values'])/count,abs_tol=1e-12)
                if count>1:assert math.isclose(s['sample_std'],math.sqrt(sum((v-s['mean'])**2 for v in s['values'])/(count-1)),abs_tol=1e-12)
                else:assert s['sample_std'] is None
    assert weight_count==2*len(m['runs'])
    assert image_count==5*sum(r['config']['variant']!='baseline' for r in m['runs'].values())
    d=json.loads((out/'wv3_band_diagnostic.json').read_text())
    assert all(len(x['rows'])==20 and len(x['mean_band_mse_peak1'])==8 for x in d['models'].values())
    print(f'PASS: {len(m["runs"])} runs, {weight_count} full weights, {image_count} fixed images, {len(manifest["artifacts"])} hashed artifacts; source/metrics verified')

if __name__=='__main__':main()
