"""Evaluate staged GF2 repeat/QB input runs and immutable references; preserve full weights."""
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
    parser.add_argument('--run-root', type=Path, default=ROOT/'experiments/a6000_gf2_repeat_qb_input')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets/a6000_gf2_repeat_qb_input')
    a = parser.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.run_root/'plan.json').read_text())
    dirs = sorted(p for p in a.run_root.iterdir() if p.is_dir() and (p/'config.json').exists())
    assert len(dirs) in (4,6)
    assert all((p/'complete.json').exists() for p in dirs)
    new_dirs = list(dirs)
    refs = json.loads((a.run_root/'references.json').read_text())
    dirs += [Path(r['path']) for r in refs]
    sensor_names = {'QB':'QuickBird', 'GF2':'Gaofen 2'}
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
    report = {'experiment':'a6000_gf2_repeat_qb_input', 'training_code_base_commit':'be02b7c',
              'training_code_changes':'next6h plan/controller/evaluator/verification scripts, source hashes preserved; commit/push on user request', 'device':'CUDA FP32 RTX A6000',
              'checkpoint_selection':'minimum validation MSE, never chosen by test', 'selection':selection, 'data':{}, 'runs':{},
              'limits':['QB test reused after earlier architecture exploration; not a new independent test claim',
                        'GF2 3 seeds; QB full/input up to3 seeds; old test sets reused, not independent confirmation', 'FR QNR resize and Q2n/SCC MATLAB parity unverified',
                        'RR PSNR fixed peak2047 per-band mean for historical comparability; GF2 normalization peak1023 also reported',
                        'old QB seed42 has no measured validation SAM or band-mean PSNR curve; no reconstruction']}
    for s, protocols in paths.items():
        report['data'][s] = {}
        for p,path in protocols.items():
            with h5py.File(path) as f: shapes = {k:list(v.shape) for k,v in f.items()}
            report['data'][s][p] = {'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path), 'shapes':shapes}
    for original,archived in [('inventory.json','training_inventory.json'),('plan.json','training_plan.json'),('references.json','training_references.json'),('validation_scores.json','validation_scores.json')]:
        shutil.copy2(a.run_root/original,out/archived)
    shutil.copy2(ROOT/'references/next6h_runs/reference_manifest.json',out/'reference_manifest.json')
    state = json.loads((a.run_root/'state.json').read_text())
    report['completed_new_runs'] = len(new_dirs)
    report['planned_new_runs'] = len(plan['runs'])
    report['deferred'] = state.get('deferred', [])
    for d in dirs:
        complete = json.loads((d/'complete.json').read_text()); assert complete['epoch'] == 100
        history = [json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in history] == list(range(1,101))
        c = complete['config']; s = c['sensor']; scale = SENSOR_MAX[s]; run_id = d.name
        r = {'config':c, 'source':str(d), 'reused':d not in new_dirs, 'history':history, 'checkpoints':{}, 'metrics':{}, 'images':[]}
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
    report['paired_summaries'] = {}
    for group,sensor,variant in [('GF2_full','GF2','interp23'),('QB_full','QB','interp23'),('QB_input','QB','interp23_input')]:
        ids = [f'{sensor}_{variant}_k6_s{s}' for s in [42,43,44] if f'{sensor}_{variant}_k6_s{s}' in report['runs']]
        records = [report['runs'][k] for k in ids]
        report['paired_summaries'][group] = {p:{key:{'mean':float(np.mean([r['delta_from_baseline'][p][key] for r in records])),
            'sample_std':float(np.std([r['delta_from_baseline'][p][key] for r in records],ddof=1)) if len(records)>1 else None,
            'seeds':[r['config']['seed'] for r in records], 'values':[r['delta_from_baseline'][p][key] for r in records]}
            for key in records[0]['metrics'][p]['mean']} for p in ['RR','FR']}
    report['QB_input_minus_full'] = {str(s):{p:{key:report['runs'][f'QB_interp23_input_k6_s{s}']['metrics'][p]['mean'][key]-report['runs'][f'QB_interp23_k6_s{s}']['metrics'][p]['mean'][key]
        for key in report['runs'][f'QB_interp23_input_k6_s{s}']['metrics'][p]['mean']} for p in ['RR','FR']} for s in [42,43,44] if f'QB_interp23_input_k6_s{s}' in report['runs']}
    write(out/'metrics.json',report)
    groups = {f'GF2_seed{s}':[f'GF2_baseline_k6_s{s}',f'GF2_interp23_k6_s{s}'] for s in [42,43,44]}
    groups.update({f'QB_input_seed{s}':[f'QB_baseline_k6_s{s}',f'QB_interp23_k6_s{s}',f'QB_interp23_input_k6_s{s}'] for s in [42,43,44] if f'QB_interp23_input_k6_s{s}' in report['runs']})
    for group,ids in groups.items():
        fig,axes = plt.subplots(1,4,figsize=(16,4))
        for run_id in ids:
            h = report['runs'][run_id]['history']
            for ax,key in zip(axes,['train_mse','val_mse','val_psnr_band_mean_peak1','val_sam_deg']):
                if all(key in x for x in h):ax.plot([x['epoch'] for x in h],[x[key] for x in h],label=run_id.replace('_k6',''))
        for ax,title in zip(axes,['Train MSE','Validation MSE','Measured validation band-mean PSNR (peak1)','Measured validation SAM (degrees)']):
            ax.set_title(title,fontsize=9); ax.set_xlabel('Epoch'); ax.grid(alpha=.25);ax.legend(fontsize=6)
        axes[0].set_yscale('log'); axes[1].set_yscale('log')
        fig.suptitle(group+' | measured values only; unavailable legacy curves omitted'); fig.tight_layout(); fig.savefig(out/f'learning_{group}.png',dpi=150); plt.close(fig)
    ids = [k for k,r in report['runs'].items() if r['config']['variant'] != 'baseline']
    fig,axes = plt.subplots(2,3,figsize=(15,9))
    for ax,(p,key,title) in zip(axes.flat,[('RR','PSNR','RR PSNR delta (up better)'),('RR','SAM','RR SAM delta (down better)'),
             ('RR','ERGAS','RR ERGAS delta (down better)'),('FR','QNR','FR QNR delta (provisional, up better)'),
             ('FR','D_lambda','FR spectral distortion delta (down better)'),('RR','MSE_peak1','RR normalized MSE delta (down better)')]):
        vals = [report['runs'][k]['delta_from_baseline'][p][key] for k in ids]
        ax.bar(range(len(ids)),vals); ax.set_xticks(range(len(ids)),[k.replace('_k6','') for k in ids],rotation=70,fontsize=7)
        ax.set_title(title); ax.axhline(0,color='black',linewidth=.8); ax.grid(axis='y',alpha=.25)
    fig.suptitle('Paired differences from same-sensor same-seed baseline'); fig.tight_layout(); fig.savefig(out/'test_metrics.png',dpi=150); plt.close(fig)
    # Exploratory diagnosis of already trained WV3 models; no new training/test selection.
    diagnostic = {'scope':'existing WV3 seed42 best only, retrospective exploratory band error analysis','models':{}}
    diagnostic_root = ROOT/'references/next6h_diagnostics'
    wv3_rr = a.data_root/'WorldView 3/Testing Dataset (ReducedData, H5 Format)/test_wv3_multiExm1.h5'
    for variant in ['baseline','interp23']:
        path=diagnostic_root/f'WV3_{variant}_k6_s42_best.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model=make_model(ck['config']);model.load_state_dict(ck['model'],strict=True);model=model.cuda().eval();rows=[]
        with h5py.File(wv3_rr) as f:
            for index in range(len(f['pan'])):
                arrays={k:np.asarray(v[index],dtype=np.float32) for k,v in f.items()}
                inputs=[torch.from_numpy(arrays[k]/2047.).unsqueeze(0).cuda() for k in ['pan','lms','ms']]
                with torch.inference_mode():pred=model(*inputs)[0].cpu().numpy().astype(np.float64)
                band_mse=np.mean((pred-arrays['gt']/2047.)**2,axis=(1,2))
                assert np.isfinite(band_mse).all()
                rows.append({'index':index,'band_mse_peak1':band_mse.tolist(),'band_psnr_peak1':(-10*np.log10(np.maximum(band_mse,1e-30))).tolist()})
        diagnostic['models'][variant]={'epoch':ck['epoch'],'weight_sha256':sha(path),'rows':rows,
            'mean_band_mse_peak1':np.mean([x['band_mse_peak1'] for x in rows],axis=0).tolist(),
            'mean_band_psnr_peak1':np.mean([x['band_psnr_peak1'] for x in rows],axis=0).tolist()}
        del model,ck;torch.cuda.empty_cache()
    diagnostic['interp23_minus_baseline']={key:(np.asarray(diagnostic['models']['interp23'][key])-np.asarray(diagnostic['models']['baseline'][key])).tolist() for key in ['mean_band_mse_peak1','mean_band_psnr_peak1']}
    write(out/'wv3_band_diagnostic.json',diagnostic)
    source_files = [ROOT/'scripts'/x for x in ['evaluate_a6000_next6h.py','evaluate_a6000.py','train_a6000_followup.py','a6000_next6h_suite.py','a6000_next_6h_plan.json','verify_a6000_next6h_report.py','run_a6000_next6h.sh']] + [ROOT/'src/ssamrn/models/pan_variants.py',ROOT/'src/ssamrn/metrics.py',ROOT/'src/ssamrn/data/pancollection.py']
    artifacts = {str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='report_manifest.json'}
    write(out/'report_manifest.json',{'artifacts':artifacts,'code':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
        'weights_format':'unaltered full best/latest checkpoint, includes Adam and RNG', 'new_runs':len(new_dirs),'planned_new_runs':6,'reused_runs':len(refs),'deferred':report['deferred'],
        'training_complete':True,'evaluation_complete':True,'selection_frozen_before_evaluation':True})
    print('COMPLETE',len(artifacts),'artifacts',flush=True)

if __name__ == '__main__': main()
