"""Experiment 55 evaluation (no training): is the WV2 transfer gain of C2MS due to the scale-mixed training or to the W2 capacity?

Arms, all WV3-trained, seeds 42-44: C0 (original) and C2MS from the 54 checkpoints, C2 (original + W2, original pairs only) from
this experiment. Metrics, operators and the v2.3 judgement are the 54 evaluate.py functions (RR, RR2, FR_WALD, FR with OBS_RMSE_DN,
sensor Wald operators, inputs normalized by 2047) on all 20 RR and FR scenes of WV3 and of WV2 (transfer).
Pairs, fixed in config.json before training: C2MS vs C2 (primary, WV2), C2 vs C0 and C2MS vs C0 (WV2 and WV3).
Also the 54 transfer-detail statistics (sharpness, detail correlation and gain, per-band bias) for the three arms, the C0/C2MS
means against the archived 54 evaluation, the C2 RR means against the controller records, WV2 images (C2 vs C2MS, seed 42) and graphs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
E54 = HERE.parent / '54_multi_sensor_scale_mixed_training'
_spec = importlib.util.spec_from_file_location('exp54_evaluate', E54 / 'evaluate.py')
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)  # also provides the 54 Model, SENSORS and Wald
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SEEDS = tuple(CFG['seeds'])
ARMS = ('C0', 'C2', 'C2MS')
BUILD = {'C0': 'C0', 'C2': 'C2MS', 'C2MS': 'C2MS'}  # C2 has the C2MS architecture (55 model.py)
TARGETS = ('WV3', 'WV2')
PAIRS = (('C2MS', 'C2'), ('C2', 'C0'), ('C2MS', 'C0'))
COLORS = {'C0': '#000000', 'C2': '#8C8C8C', 'C2MS': '#C44E52'}


def checkpoint(arm, seed, runtime):
    if arm == 'C2':
        archived = HERE / f'checkpoints/C2_WV3_s{seed}/best.pt'
        return archived if archived.exists() else runtime / f'runs/WV3_C2_s{seed}/checkpoints/best.pt'
    return E54 / f'checkpoints/{arm}_WV3_s{seed}/best.pt'


def history(arm, seed, runtime):
    if arm == 'C2':
        archived = HERE / f'results/C2_WV3_s{seed}/history.json'
        return archived if archived.exists() else runtime / f'runs/WV3_C2_s{seed}/history.json'
    return E54 / f'results/{arm}_WV3_s{seed}/history.json'


def detail(pred, a, hf):
    dp, dg = (pred - a['lms']).ravel(), (a['gt'] - a['lms']).ravel()
    return dict(sharpness=hf(pred) / hf(a['gt']), detail_corr=float(np.corrcoef(dp, dg)[0, 1]), detail_gain=float(dp.std() / dg.std()),
                bias=((pred - a['gt']).mean(axis=(1, 2)) / a['gt'].mean(axis=(1, 2))).tolist())


def graphs(models, judgements, runtime, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for arm in ARMS:
        for n, s in enumerate(SEEDS):
            p = history(arm, s, runtime)
            if not p.exists(): continue
            h = [r for r in json.loads(p.read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
            kw = dict(color=COLORS[arm], alpha=1 if s == 42 else .45, lw=1.4 if s == 42 else .9, label=arm if n == 0 else None)
            ax[0].plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], **kw)
            c = [r for r in h if r.get('coarse_val_psnr') is not None]
            ax[1].plot([r['epoch'] for r in c], [r['coarse_val_psnr'] for r in c], **kw)
    ax[0].set(xlabel='epoch', ylabel='PSNR (dB, peak 1)', title='WV3 validation, original scale (selection)', ylim=(25, None))
    ax[1].set(xlabel='epoch', ylabel='PSNR (dB, peak 1)', title='WV3 validation, 2x coarser pairs (logged only)', ylim=(25, None))
    for x in ax: x.legend(fontsize=8); x.grid(alpha=.3)
    fig.suptitle('Learning curves, seed 42 bold and seeds 43-44 light (C0 and C2MS from 54, C2 from 55)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=130); plt.close(fig)
    tests = ('RR', 'RR2', 'FR_WALD'); x = np.arange(len(tests)); w = .26
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, target in zip(ax, TARGETS):
        for j, (cand, ctrl) in enumerate(PAIRS):
            key = f'{target}:{cand}:{ctrl}@' + '+'.join(map(str, SEEDS))
            v = [judgements[key]['delta'][t]['PSNR'] for t in tests]
            per = [[judgements[f'{target}:{cand}:{ctrl}@{s}']['delta'][t]['PSNR'] for s in SEEDS] for t in tests]
            a.bar(x + (j - 1) * w, v, w, color=['#C44E52', '#8C8C8C', '#4C72B0'][j], label=f'{cand} - {ctrl}')
            for xi, d in zip(x + (j - 1) * w, per): a.scatter([xi] * len(d), d, color='black', s=8, zorder=3)
        a.axhline(0, color='black', lw=.8); a.set_xticks(x, tests); a.grid(axis='y', alpha=.3)
        a.set_title(target + (' (WV3-trained, transfer)' if target == 'WV2' else ' (training sensor)'), fontsize=10)
    ax[0].set_ylabel('PSNR difference, same seed (dB)'); ax[1].legend(fontsize=8)
    fig.suptitle('Mean of 20 test scenes; bars = 3-seed mean, dots = individual seeds', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(CFG['runtime']), help='55 runtime with runs/WV3_C2_s<seed>/ (until the C2 weights are archived)')
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
    lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float64).view(1, 1, 3, 3)
    def hf(x): t = torch.from_numpy(x)[:, None]; return float(torch.nn.functional.conv2d(torch.nn.functional.pad(t, (1, 1, 1, 1), mode='replicate'), lap).abs().mean())
    nets, models = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            path = checkpoint(arm, s, a.runtime); ck = torch.load(path, map_location='cpu', weights_only=False)
            if (ck.get('sensor'), ck.get('arm'), ck.get('seed')) != ('WV3', arm, s): raise ValueError(f'checkpoint identity {path}')
            net = V.Model(BUILD[arm], 8, s); net.load_state_dict(ck['model'], strict=True); nets[f'{arm}_s{s}'] = net.to(a.device).eval()
            models[f'{arm}_s{s}'] = dict(checkpoint=str(path), checkpoint_sha256=V.sha(path), best_epoch=ck['best_epoch'], targets={})
    checks, data_sha = {}, {}
    for target in TARGETS:
        scenes = V.load(a.data, target, a.limit); data_sha[target] = {k: scenes[k + '_sha256'] for k in ('RR', 'FR')}
        checks[f'{target}_operators'] = V.operator_check(a.data, target, scenes, a.device); wald = V.Wald(target, a.device)
        for key, net in nets.items():
            rows, det = [], []
            for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR'])):
                rows.append({'index': i, **V.scene_metrics(net, rr, fr, wald, a.device)})
                det.append(detail(V.predict(net, rr['pan'], rr['lms'], rr['ms'], 2047., a.device), rr, hf))
            mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
            dmean = {k: float(np.mean([d[k] for d in det])) for k in ('sharpness', 'detail_corr', 'detail_gain')}
            dmean['band_bias'] = np.mean([d['bias'] for d in det], axis=0).tolist(); dmean['max_abs_band_bias'] = float(np.max(np.abs(dmean['band_bias'])))
            models[key]['targets'][target] = dict(mean=mean, detail=dmean, rows=rows)
            print(target, key, 'RR %.4f/%.4f/%.4f RR2 %.4f FR_WALD %.4f OBS %.3f sharp %.3f bias %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'],
                  mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], dmean['sharpness'], dmean['max_abs_band_bias']), flush=True)
        del scenes
    judgements = {}
    for target in TARGETS:
        for cand, ctrl in PAIRS:
            for seeds in (SEEDS,) + tuple((s,) for s in SEEDS):
                name = f'{target}:{cand}:{ctrl}@' + '+'.join(map(str, seeds))
                c = [models[f'{cand}_s{s}']['targets'][target] for s in seeds]; b = [models[f'{ctrl}_s{s}']['targets'][target] for s in seeds]
                j = V.judge([x['mean'] for x in c], [x['mean'] for x in b]); j['seeds'] = list(seeds); j['scenes_improved'] = V.scene_wins([x['rows'] for x in c], [x['rows'] for x in b])
                judgements[name] = j
                print(name, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d' % (
                    j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
                    j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR']), flush=True)
    three = '+'.join(map(str, SEEDS))
    prefixed = {'primary_WV2_C2MS_vs_C2': judgements[f'WV2:C2MS:C2@{three}']['label'], 'WV2_C2_vs_C0': judgements[f'WV2:C2:C0@{three}']['label'],
                'WV3_C2MS_vs_C2': judgements[f'WV3:C2MS:C2@{three}']['label'], 'WV3_C2_vs_C0': judgements[f'WV3:C2:C0@{three}']['label']}
    consistency = {}
    if a.limit is None:
        ref54 = json.loads((E54 / 'results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models']
        for arm in ('C0', 'C2MS'):
            for s in SEEDS:
                for target in TARGETS:
                    r = ref54[f'{arm}_WV3_s{s}']['targets'][target]['mean']; m = models[f'{arm}_s{s}']['targets'][target]['mean']
                    consistency[f'{arm}_s{s}:{target}:vs_54'] = max(abs(m[part][k] - r[part][k]) for part in ('RR', 'RR2', 'FR_WALD') for k in ('PSNR', 'SAM', 'ERGAS'))
        for s in SEEDS:
            for target, name in (('WV3', 'RR'), ('WV2', 'WV2_RR')):
                rec = a.runtime / f'runs/WV3_C2_s{s}/evaluation/{name}.json'
                if not rec.exists(): rec = HERE / f'results/C2_WV3_s{s}/evaluation/{name}.json'
                if rec.exists():
                    ref = json.loads(rec.read_text(encoding='utf-8'))['mean']
                    consistency[f'C2_s{s}:{target}:vs_controller'] = max(abs(models[f'C2_s{s}']['targets'][target]['mean']['RR'][k] - ref[k]) for k in ('PSNR', 'SAM', 'ERGAS'))
    print('pre-fixed', prefixed, 'max consistency diff', max(consistency.values(), default=None), flush=True)
    t = V.SENSORS['WV2']
    V.result_images.render(nets['C2_s42'], nets['C2MS_s42'], a.data, out / 'images/WV2_C2MS_vs_C2', names=('C2', 'C2MS'), seed=42, device=a.device,
                           scenes=tuple(i for i in CFG['scenes'] if a.limit is None or i < a.limit), labels=('C2 (W2만, 축척 혼합 없음)', 'C2MS (현재 기준)'),
                           rr_file=V.files('WV2')['RR'], scale=t['max_value'], bands=t['rgb'], land_cover={})
    graphs(models, judgements, a.runtime, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3 with the 54 sensor Wald operators', noise=V.NOISE, tolerances=V.TOL, seeds=list(SEEDS), arms=list(ARMS), pairs=PAIRS,
        scenes_per_target=a.limit or 20, prefixed_judgement=prefixed, judgements=judgements, checks=checks, consistency=consistency, models=models,
        detail_definition=V.transfer_detail.__doc__,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=V.sha(__file__),
                         evaluate54_sha256=V.sha(E54 / 'evaluate.py'), result_images_sha256=V.sha(V.ROOT / 'common/result_images.py'), data_sha256=data_sha)),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
