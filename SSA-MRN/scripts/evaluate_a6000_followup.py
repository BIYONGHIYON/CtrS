"""Evaluate completed follow-up and frozen QB seed42 references; preserve full weights."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssamrn.models.pan_variants import make_model
from ssamrn.metrics import rr_metrics, fr_metrics
from ssamrn.data.pancollection import SENSOR_MAX
from evaluate_a6000 import panels, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT/'experiments/a6000_followup_123')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets/a6000_followup_123')
    a = parser.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.run_root/'plan.json').read_text())
    dirs = sorted(p for p in a.run_root.iterdir() if p.is_dir() and (p/'config.json').exists())
    assert len(dirs) == 10
    refs = json.loads((a.run_root/'references.json').read_text())
    dirs += [Path(r['path']) for r in refs]
    sensor_names = {'QB':'QuickBird', 'GF2':'Gaofen 2', 'WV3':'WorldView 3'}
    paths = {s:{'RR':a.data_root/n/'Testing Dataset (ReducedData, H5 Format)'/f'test_{s.lower()}_multiExm1.h5',
               'FR':a.data_root/n/'Testing Dataset (FullData, H5 Format)'/f'test_{s.lower()}_OrigScale_multiExm1.h5'} for s,n in sensor_names.items()}
    # Freeze all sensors before loading checkpoints or observing test metrics.
    selection = {}
    for s in paths:
        with h5py.File(paths[s]['RR']) as f: count = len(f['pan'])
        selection[s] = {'seed':42, 'protocol':'RR', 'zero_based_scene_indices':sorted(np.random.default_rng(42).choice(count,5,replace=False).tolist()),
                        'scene_id':'H5 index; original acquisition IDs unavailable', 'tile':'full scene', 'order':'ascending H5 index',
                        'rule':'fixed before this test evaluation; QB test previously used in architecture exploration'}
    if (out/'selection.json').exists(): assert json.loads((out/'selection.json').read_text()) == selection
    else: write(out/'selection.json', selection)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    report = {'experiment':'a6000_followup_123', 'training_code_base_commit':'c5e92796bd0ade9518bdd663dd2fe499161d4a2d',
              'training_code_changes':'follow-up scripts and pan_variants included in delivery commit', 'device':'CUDA FP32 RTX A6000',
              'checkpoint_selection':'minimum validation MSE, never chosen by test', 'selection':selection, 'data':{}, 'runs':{},
              'limits':['QB test reused after earlier architecture exploration; not a new independent test claim',
                        'GF2/WV3 single seed only', 'FR QNR resize and Q2n/SCC MATLAB parity unverified',
                        'RR PSNR fixed peak2047 per-band mean for historical comparability; GF2 normalization peak1023 also reported',
                        'old QB seed42 has no measured validation SAM or band-mean PSNR curve; no reconstruction']}
    for s, protocols in paths.items():
        report['data'][s] = {}
        for p,path in protocols.items():
            with h5py.File(path) as f: shapes = {k:list(v.shape) for k,v in f.items()}
            report['data'][s][p] = {'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path), 'shapes':shapes}
    for d in dirs:
        complete = json.loads((d/'complete.json').read_text()); assert complete['epoch'] == 100
        history = [json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in history] == list(range(1,101))
        c = complete['config']; s = c['sensor']; scale = SENSOR_MAX[s]; run_id = d.name
        r = {'config':c, 'source':str(d), 'reused':d not in dirs[:10], 'history':history, 'checkpoints':{}, 'metrics':{}, 'images':[]}
        for kind in ['best','latest']:
            source = d/f'{kind}.pt'; ck = torch.load(source, map_location='cpu', weights_only=True)
            assert ck['config'] == c
            if kind == 'latest': assert ck['epoch'] == 100
            else:
                assert ck['epoch'] == min(history, key=lambda x:x['val_mse'])['epoch']
                assert np.isclose(ck['best_mse'], min(x['val_mse'] for x in history), rtol=1e-10)
                best = ck
            dest = out/'weights'/run_id/f'{kind}.pt'; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,dest)
            assert sha(source) == sha(dest)
            r['checkpoints'][kind] = {'epoch':ck['epoch'], 'best_mse':ck['best_mse'], 'file':str(dest.relative_to(out)),
                                      'bytes':dest.stat().st_size, 'sha256':sha(dest), 'format':'full model/config/Adam/RNG resume checkpoint'}
        model = make_model(c); model.load_state_dict(best['model'], strict=True); model = model.cuda().eval()
        for p,path in paths[s].items():
            rows = []; started = time.monotonic()
            with h5py.File(path) as f:
                for index in range(len(f['pan'])):
                    arrays = {k:np.asarray(v[index],dtype=np.float32) for k,v in f.items()}
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    inputs = [torch.from_numpy(arrays[k]/scale).unsqueeze(0).cuda() for k in ['pan','lms','ms']]
                    with torch.inference_mode(): prediction = model(*inputs)[0].cpu().numpy().astype(np.float64)*scale
                    assert np.isfinite(prediction).all()
                    if p == 'RR':
                        metrics = rr_metrics(arrays['gt'], prediction)
                        metrics['MSE_peak1'] = float(np.mean(((prediction-arrays['gt'])/scale)**2))
                        metrics['PSNR_sensor_peak'] = metrics['PSNR'] + float(20*np.log10(scale/2047.))
                        if c['variant'] != 'baseline' and index in selection[s]['zero_based_scene_indices']:
                            name = f'{run_id}_scene_{index:02}.png'
                            display = panels(arrays,prediction,out/name,f'{s} {c["variant"]} s{c["seed"]}',index)
                            # Replace historical helper's QB caption with the actual sensor.
                            from PIL import Image, ImageDraw
                            im = Image.open(out/name); draw = ImageDraw.Draw(im); draw.rectangle((0,21,im.width,37), fill='white')
                            draw.text((8,23),f'{s} RR index {index} | full scene | shared GT 1-99% display stretch',fill='black'); im.save(out/name)
                            r['images'].append({'index':index,'file':name,'display':display})
                    else: metrics = fr_metrics(prediction,arrays['lms'],arrays['ms'],arrays['pan'])
                    assert all(np.isfinite(v) for v in metrics.values())
                    rows.append({'index':index,'metrics':metrics})
            r['metrics'][p] = {'count':len(rows), 'mean':{k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']},
                               'rows':rows, 'seconds':time.monotonic()-started}
            print(run_id,p,json.dumps(r['metrics'][p]['mean']),flush=True)
        report['runs'][run_id] = r; write(out/'metrics.json',report)
        del model, best, ck; torch.cuda.empty_cache()
    for run_id,r in report['runs'].items():
        base = report['runs'][f'{r["config"]["sensor"]}_baseline_k6_s{r["config"]["seed"]}']
        r['delta_from_baseline'] = {p:{k:v-base['metrics'][p]['mean'][k] for k,v in r['metrics'][p]['mean'].items()} for p in ['RR','FR']}
    qb = [report['runs'][f'QB_interp23_k6_s{x}'] for x in [42,43,44]]
    report['QB_paired_deltas_summary'] = {p:{k:{'mean':float(np.mean([r['delta_from_baseline'][p][k] for r in qb])),
         'sample_std':float(np.std([r['delta_from_baseline'][p][k] for r in qb],ddof=1)),
         'seed_order':[42,43,44], 'values':[r['delta_from_baseline'][p][k] for r in qb]} for k in qb[0]['metrics'][p]['mean']} for p in ['RR','FR']}
    write(out/'metrics.json',report)
    groups = {'QB_seed43':['QB_baseline_k6_s43','QB_interp23_k6_s43'], 'QB_seed44':['QB_baseline_k6_s44','QB_interp23_k6_s44'],
              'QB_position':['QB_interp23_input_k6_s42','QB_interp23_output_k6_s42'],
              'GF2':['GF2_baseline_k6_s42','GF2_interp23_k6_s42'], 'WV3':['WV3_baseline_k6_s42','WV3_interp23_k6_s42']}
    for group,ids in groups.items():
        fig,axes = plt.subplots(1,4,figsize=(16,4))
        for run_id in ids:
            h = report['runs'][run_id]['history']
            for ax,key in zip(axes,['train_mse','val_mse','val_psnr_band_mean_peak1','val_sam_deg']): ax.plot([x['epoch'] for x in h],[x[key] for x in h],label=run_id.replace('_k6',''))
        for ax,title in zip(axes,['Train MSE','Validation MSE','Validation band-mean PSNR (peak1)','Validation SAM (degrees)']):
            ax.set_title(title,fontsize=10); ax.set_xlabel('Epoch'); ax.grid(alpha=.25)
        axes[0].set_yscale('log'); axes[1].set_yscale('log'); axes[3].legend(fontsize=6)
        fig.suptitle(group+' | actual measured values'); fig.tight_layout(); fig.savefig(out/f'learning_{group}.png',dpi=150); plt.close(fig)
    ids = [k for k,r in report['runs'].items() if r['config']['variant'] != 'baseline']
    fig,axes = plt.subplots(2,3,figsize=(15,9))
    for ax,(p,key,title) in zip(axes.flat,[('RR','PSNR','RR PSNR delta (up better)'),('RR','SAM','RR SAM delta (down better)'),
             ('RR','ERGAS','RR ERGAS delta (down better)'),('FR','QNR','FR QNR delta (provisional, up better)'),
             ('FR','D_lambda','FR spectral distortion delta (down better)'),('RR','MSE_peak1','RR normalized MSE delta (down better)')]):
        vals = [report['runs'][k]['delta_from_baseline'][p][key] for k in ids]
        ax.bar(range(len(ids)),vals); ax.set_xticks(range(len(ids)),[k.replace('_k6','') for k in ids],rotation=70,fontsize=7)
        ax.set_title(title); ax.axhline(0,color='black',linewidth=.8); ax.grid(axis='y',alpha=.25)
    fig.suptitle('Paired differences from same-sensor same-seed baseline'); fig.tight_layout(); fig.savefig(out/'test_metrics.png',dpi=150); plt.close(fig)
    source_files = [ROOT/'scripts'/x for x in ['evaluate_a6000_followup.py','evaluate_a6000.py','train_a6000_followup.py','a6000_followup_suite.py','a6000_followup_plan.json']] + [ROOT/'src/ssamrn/models/pan_variants.py', ROOT/'src/ssamrn/metrics.py',ROOT/'src/ssamrn/data/pancollection.py']
    artifacts = {str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report_manifest.json'}
    write(out/'report_manifest.json',{'artifacts':artifacts,'code':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
        'weights_format':'unaltered full best/latest checkpoint, includes Adam and RNG', 'new_runs':10,'reused_runs':2,
        'training_complete':True,'evaluation_complete':True,'selection_frozen_before_evaluation':True})
    print('COMPLETE',len(artifacts),'artifacts',flush=True)

if __name__ == '__main__': main()
