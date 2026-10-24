"""Experiment 58 evaluation (no training): C2MS + spectral loss (C2MS_SAM) vs C2MS, seeds 42-44, on QB, GF2, WV3 and WV2 (transfer).

C2MS_SAM: QB seed 42 from the 57 run, all other runs from this experiment. Controls of the same seeds: QB from 53, GF2 and WV3
from 54 (WV2 = the WV3-trained models). All targets use the 54 evaluate.py functions: sensor Wald operator for RR2, FR_WALD and
the FR observation (equal to common/evaluation_protocol on QB, checked in 54), inputs normalized by the sensor maximum, and the
v2.3 judgement on same-seed paired means (noise threshold 0.04 dB). Also the EMA weights of C2MS_SAM, the sharpness / detail /
colour statistics of the 54 transfer analysis, consistency against the archived 53/54/56 evaluations and the controller records,
QB and WV2 images (C2MS vs C2MS_SAM, seed 42) and graphs. Writes only below results/ (or --out).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, importlib.util, json, sys
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
_spec = importlib.util.spec_from_file_location('exp54_evaluate', PERF / '54_multi_sensor_scale_mixed_training/evaluate.py')
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)  # predict, scene_metrics, judge, scene_wins, operator_check, sha, Model, SENSORS, Wald
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SEEDS = tuple(CFG['seeds'])
TARGETS = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}  # test target -> training sensor
CONTROL = {'QB': PERF / '53_scale_mixed_training', 'GF2': PERF / '54_multi_sensor_scale_mixed_training', 'WV3': PERF / '54_multi_sensor_scale_mixed_training'}
E57 = PERF / '57_c2ms_training_conditions'
COLORS = {'C2MS': '#000000', 'C2MS_SAM': '#C44E52', 'C2MS_SAM_EMA': '#E39A9C'}
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float64).view(1, 1, 3, 3)


def candidate(sensor, seed, runtime, fname='best.pt'):
    if sensor == 'QB' and seed == 42: return E57 / f'checkpoints/C2MS_SAM_QB_s42/{fname}'
    archived = HERE / f'checkpoints/C2MS_SAM_{sensor}_s{seed}/{fname}'
    return archived if archived.exists() else runtime / f'runs/{sensor}_C2MS_SAM_s{seed}/checkpoints/{fname}'


def control(sensor, seed): return CONTROL[sensor] / f'checkpoints/C2MS_{sensor}_s{seed}/best.pt'


def history(arm, sensor, seed, runtime):
    if arm == 'C2MS': return CONTROL[sensor] / f'results/C2MS_{sensor}_s{seed}/history.json'
    if sensor == 'QB' and seed == 42: return E57 / 'results/C2MS_SAM_QB_s42/history.json'
    archived = HERE / f'results/C2MS_SAM_{sensor}_s{seed}/history.json'
    return archived if archived.exists() else runtime / f'runs/{sensor}_C2MS_SAM_s{seed}/history.json'


def load(data, target, limit):
    table = CFG['data_sha256'] if target in CFG['data_sha256'] else CFG['transfer_data_sha256']; scenes = {}
    for proto, name in V.files(target).items():
        found = list(data.rglob(name)); assert len(found) == 1, name
        digest = V.sha(found[0])
        if digest != table[target][proto]: raise ValueError(f'data sha mismatch {target} {proto}')
        with h5py.File(found[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if proto == 'RR' else ('pan', 'lms', 'ms')
            n = len(f['pan']) if limit is None else limit
            scenes[proto] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(n)]
        scenes[proto + '_sha256'] = digest
    return scenes


def detail(pred, a):
    def hf(x): t = torch.from_numpy(x)[:, None]; return float(torch.nn.functional.conv2d(torch.nn.functional.pad(t, (1, 1, 1, 1), mode='replicate'), LAP).abs().mean())
    dp, dg = (pred - a['lms']).ravel(), (a['gt'] - a['lms']).ravel()
    return dict(sharpness=hf(pred) / hf(a['gt']), detail_corr=float(np.corrcoef(dp, dg)[0, 1]), detail_gain=float(dp.std() / dg.std()),
                bias=((pred - a['gt']).mean(axis=(1, 2)) / a['gt'].mean(axis=(1, 2))).tolist())


def graphs(models, judgements, runtime, out):
    learning_metrics(runtime, out)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for a, sensor in zip(ax, ('QB', 'GF2', 'WV3')):
        lo = []
        for arm in ('C2MS', 'C2MS_SAM'):
            for n, s in enumerate(SEEDS):
                p = history(arm, sensor, s, runtime)
                if not p.exists(): continue
                h = [r for r in json.loads(p.read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
                a.plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], color=COLORS[arm], alpha=1 if s == 42 else .45, lw=1.4 if s == 42 else .9, label=arm if n == 0 else None)
                lo.append(h[-1]['val_psnr'])
        top = max(lo) if lo else 0
        a.set(xlabel='epoch', ylabel='PSNR (dB)', title=f'{sensor} validation, original scale (selection)', ylim=(top - 1.5, top + 0.2)); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle('Learning curves, seed 42 bold and seeds 43-44 light (C2MS from 53/54; C2MS_SAM QB seed 42 from 57, others from 58)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2)); tests = ('RR', 'RR2', 'FR_WALD'); x = np.arange(len(tests)); three = '+'.join(map(str, SEEDS))
    for a, target in zip(ax, TARGETS):
        j = judgements[f'{target}:C2MS_SAM:C2MS@{three}']; v = [j['delta'][t]['PSNR'] for t in tests]
        per = [[judgements[f'{target}:C2MS_SAM:C2MS@{s}']['delta'][t]['PSNR'] for s in SEEDS] for t in tests]
        a.bar(x, v, .55, color=COLORS['C2MS_SAM'])
        for xi, d in zip(x, per): a.scatter([xi] * len(d), d, color='black', s=9, zorder=3)
        for xi, vi in zip(x, v): a.annotate(f'{vi:+.3f}', (xi, vi), xytext=(0, 4 if vi >= 0 else -4), textcoords='offset points', ha='center', va='bottom' if vi >= 0 else 'top', fontsize=8)
        a.axhline(0, color='black', lw=.8); a.axhline(0.04, color='#888888', lw=.8, ls=':'); a.set_xticks(x, tests); a.grid(axis='y', alpha=.3)
        a.set_title(target + (' (WV3-trained, transfer)' if target == 'WV2' else ''), fontsize=10)
    ax[0].set_ylabel('PSNR difference C2MS_SAM - C2MS, same seed (dB)')
    fig.suptitle('Mean of 20 test scenes; bars = 3-seed mean, dots = seeds, dotted = 3-seed noise threshold 0.04 dB', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def learning_metrics(runtime, out):
    """Plot retained histories only; no model inference or training."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for col, sensor in enumerate(('QB', 'GF2', 'WV3')):
        for arm in ('C2MS', 'C2MS_SAM'):
            for seed in SEEDS:
                path = history(arm, sensor, seed, runtime)
                records = json.loads(path.read_text(encoding='utf-8'))
                for row, (key, label) in enumerate((('train_mse', 'Training MSE (original pairs)'), ('val_psnr', 'Validation PSNR (dB)'), ('val_sam', 'Validation SAM (degrees)'))):
                    measured = [r for r in records if r.get(key) is not None]
                    axes[row, col].plot([r['epoch'] for r in measured], [r[key] for r in measured], color=COLORS[arm], alpha=1 if seed == 42 else .4,
                                        lw=1.4 if seed == 42 else .8, label=arm if seed == 42 else None)
                    axes[row, col].set_ylabel(label); axes[row, col].grid(alpha=.25)
                    if row in (0, 2): axes[row, col].set_yscale('log')
        axes[0, col].set_title(sensor); axes[0, col].legend(fontsize=8)
        axes[2, col].set_xlabel('Epoch')
    fig.suptitle('Measured histories: seed 42 bold, seeds 43-44 light. Controls: 53/54; QB candidate seed 42: 57.', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'learning_metrics.png', dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(CFG['runtime']))
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='first N scenes only (code check; the judgement needs all 20)')
    p.add_argument('--graphs-only', action='store_true')
    a = p.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    if a.graphs_only:
        r = json.loads((out / 'evaluation_v2_3.json').read_text(encoding='utf-8')); return graphs(r['models'], r['judgements'], a.runtime, out)
    if (out / 'evaluation_v2_3.json').exists(): raise FileExistsError(out / 'evaluation_v2_3.json')
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    nets, models = {}, {}
    for sensor in ('QB', 'GF2', 'WV3'):
        bands = V.SENSORS[sensor]['bands']
        for s in SEEDS:
            for arm, path in (('C2MS', control(sensor, s)), ('C2MS_SAM', candidate(sensor, s, a.runtime)), ('C2MS_SAM_EMA', candidate(sensor, s, a.runtime, 'best_ema.pt'))):
                ck = torch.load(path, map_location='cpu', weights_only=False)
                if arm != 'C2MS' and (ck.get('sensor'), ck.get('arm'), ck.get('seed')) != (sensor, 'C2MS_SAM', s): raise ValueError(f'checkpoint identity {path}')
                net = V.Model('C2MS', bands, s); net.load_state_dict(ck['model'], strict=True); nets[f'{arm}_{sensor}_s{s}'] = net.to(a.device).eval()
                models[f'{arm}_{sensor}_s{s}'] = dict(checkpoint=str(path), checkpoint_sha256=V.sha(path), best_epoch=ck.get('best_epoch'), targets={})
    checks, data_sha = {}, {}
    for target, sensor in TARGETS.items():
        scenes = load(a.data, target, a.limit); data_sha[target] = {k: scenes[k + '_sha256'] for k in ('RR', 'FR')}
        checks[f'{target}_operators'] = V.operator_check(a.data, target, scenes, a.device); wald = V.Wald(target, a.device); scale = V.SENSORS[target]['max_value']
        for arm in ('C2MS', 'C2MS_SAM', 'C2MS_SAM_EMA'):
            for s in SEEDS:
                key = f'{arm}_{sensor}_s{s}'; net = nets[key]; rows, det = [], []
                for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR'])):
                    rows.append({'index': i, **V.scene_metrics(net, rr, fr, wald, a.device)}); det.append(detail(V.predict(net, rr['pan'], rr['lms'], rr['ms'], scale, a.device), rr))
                mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
                dm = {k: float(np.mean([d[k] for d in det])) for k in ('sharpness', 'detail_corr', 'detail_gain')}
                dm['band_bias'] = np.mean([d['bias'] for d in det], axis=0).tolist(); dm['max_abs_band_bias'] = float(np.max(np.abs(dm['band_bias'])))
                models[key]['targets'][target] = dict(mean=mean, detail=dm, rows=rows)
                print(target, key, 'RR %.4f/%.4f/%.4f RR2 %.4f FR_WALD %.4f OBS %.3f sharp %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'], mean['RR2']['PSNR'],
                      mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], dm['sharpness']), flush=True)
        del scenes
    judgements = {}
    for target, sensor in TARGETS.items():
        for cand in ('C2MS_SAM', 'C2MS_SAM_EMA'):
            for seeds in (SEEDS,) + tuple((s,) for s in SEEDS):
                name = f'{target}:{cand}:C2MS@' + '+'.join(map(str, seeds))
                c = [models[f'{cand}_{sensor}_s{s}']['targets'][target] for s in seeds]; b = [models[f'C2MS_{sensor}_s{s}']['targets'][target] for s in seeds]
                j = V.judge([x['mean'] for x in c], [x['mean'] for x in b]); j['seeds'] = list(seeds); j['scenes_improved'] = V.scene_wins([x['rows'] for x in c], [x['rows'] for x in b])
                judgements[name] = j
                print(name, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d/%d' % (
                    j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
                    j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR'], j['scenes_improved']['RR']['SAM']), flush=True)
    three = '+'.join(map(str, SEEDS))
    prefixed = {f'{t}_C2MS_SAM_vs_C2MS': judgements[f'{t}:C2MS_SAM:C2MS@{three}']['label'] for t in TARGETS}
    consistency = {}
    if a.limit is None:
        ref56 = json.loads((PERF / '56_original_scale_mixed_training/results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models']
        ref54 = json.loads((PERF / '54_multi_sensor_scale_mixed_training/results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models']
        for s in SEEDS:
            r = ref56[f'C2MS_s{s}']['mean']; m = models[f'C2MS_QB_s{s}']['targets']['QB']['mean']
            consistency[f'C2MS_QB_s{s}:vs_56'] = max(abs(m[part][k] - r[part][k]) for part in ('RR', 'RR2', 'FR_WALD') for k in ('PSNR', 'SAM', 'ERGAS'))
            for sensor, targets in (('GF2', ('GF2',)), ('WV3', ('WV3', 'WV2'))):
                for t in targets:
                    r = ref54[f'C2MS_{sensor}_s{s}']['targets'][t]['mean']; m = models[f'C2MS_{sensor}_s{s}']['targets'][t]['mean']
                    consistency[f'C2MS_{sensor}_s{s}:{t}:vs_54'] = max(abs(m[part][k] - r[part][k]) for part in ('RR', 'RR2', 'FR_WALD') for k in ('PSNR', 'SAM', 'ERGAS'))
        for target, sensor in TARGETS.items():
            for s in SEEDS:
                name = 'RR' if target == sensor else f'{target}_RR'
                rec = (E57 / 'results/C2MS_SAM_QB_s42' if (sensor, s) == ('QB', 42) else (HERE / f'results/C2MS_SAM_{sensor}_s{s}' if (HERE / f'results/C2MS_SAM_{sensor}_s{s}').exists() else a.runtime / f'runs/{sensor}_C2MS_SAM_s{s}')) / f'evaluation/{name}.json'
                if rec.exists():
                    r = json.loads(rec.read_text(encoding='utf-8'))['mean']
                    consistency[f'C2MS_SAM_{sensor}_s{s}:{target}:vs_controller'] = max(abs(models[f'C2MS_SAM_{sensor}_s{s}']['targets'][target]['mean']['RR'][k] - r[k]) for k in ('PSNR', 'SAM', 'ERGAS'))
    print('pre-fixed', prefixed, 'max consistency diff', max(consistency.values(), default=None), flush=True)
    for target, sensor in (('QB', 'QB'), ('WV2', 'WV3')):
        t = V.SENSORS[target]
        V.result_images.render(nets[f'C2MS_{sensor}_s42'], nets[f'C2MS_SAM_{sensor}_s42'], a.data, out / f'images/{target}_C2MS_SAM_vs_C2MS', names=('C2MS', 'C2MS+분광'), seed=42,
                               device=a.device, scenes=tuple(i for i in CFG['scenes'] if a.limit is None or i < a.limit), labels=('현재 기준 C2MS', 'C2MS + 분광 손실'),
                               rr_file=V.files(target)['RR'], scale=t['max_value'], bands=t['rgb'], land_cover=V.result_images.LAND_COVER if target == 'QB' else {})
    graphs(models, judgements, a.runtime, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3 with the 54 sensor Wald operators', noise=V.NOISE, tolerances=V.TOL, seeds=list(SEEDS), targets=TARGETS,
        reused_runs=CFG['reused_runs'], scenes_per_target=a.limit or 20, prefixed_judgement=prefixed, judgements=judgements, checks=checks, consistency=consistency, models=models,
        detail_definition=V.transfer_detail.__doc__,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=V.sha(__file__),
                         evaluate54_sha256=V.sha(PERF / '54_multi_sensor_scale_mixed_training/evaluate.py'), result_images_sha256=V.sha(V.ROOT / 'common/result_images.py'), data_sha256=data_sha)),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
