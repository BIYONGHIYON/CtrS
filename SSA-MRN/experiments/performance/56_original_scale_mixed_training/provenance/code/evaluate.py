"""Experiment 56 evaluation (no training): the QB 2x2 of W2 capacity x scale-mixed training.

Arms, QB, seeds 42-44: C0 and C2 from the 43 checkpoints, C2MS from the 53 checkpoints, C0MS (original + scale-mixed training)
from this experiment. All are measured with common/evaluation_protocol.scene_metrics on all 20 RR and FR test scenes and judged
with the 53 evaluate.py v2.3 functions (same-seed pairs, seed means). Pairs fixed in config.json before training: C0MS vs C0
(primary), C2 vs C0, C2MS vs C2, C2MS vs C0MS, C2MS vs C0. Interaction = (C2MS - C2) - (C0MS - C0) of the seed-paired means.
Also the C0/C2/C2MS means against the archived 53 evaluation, the C0MS RR means against the controller records, QB images
(C0 vs C0MS, seed 42) and graphs. Writes only below results/ (or --out).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, importlib.util, json, sys
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
E53 = HERE.parent / '53_scale_mixed_training'
_spec = importlib.util.spec_from_file_location('exp53_evaluate', E53 / 'evaluate.py')
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)  # judge, scene_wins, sha, protocol, result_images, M53 (48 models)
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SEEDS = tuple(CFG['seeds'])
ARMS = ('C0', 'C0MS', 'C2', 'C2MS')
BUILD = {'C0': 'C0', 'C0MS': 'C0', 'C2': 'C2', 'C2MS': 'C2'}
PAIRS = (('C0MS', 'C0'), ('C2', 'C0'), ('C2MS', 'C2'), ('C2MS', 'C0MS'), ('C2MS', 'C0'))
PARTS = ('RR', 'RR2', 'FR_WALD')
COLORS = {'C0': '#000000', 'C0MS': '#55A868', 'C2': '#8C8C8C', 'C2MS': '#C44E52'}


def checkpoint(arm, seed, runtime):
    if arm == 'C0MS':
        archived = HERE / f'checkpoints/C0MS_QB_s{seed}/best.pt'
        return archived if archived.exists() else runtime / f'runs/QB_C0MS_s{seed}/checkpoints/best.pt'
    return V.ckpt(arm, seed)


def history(arm, seed, runtime):
    if arm == 'C0MS':
        archived = HERE / f'results/C0MS_QB_s{seed}/history.json'
        return archived if archived.exists() else runtime / f'runs/QB_C0MS_s{seed}/history.json'
    return V.history(arm, seed)


def interaction(models, seeds):
    """(C2MS - C2) - (C0MS - C0) per test and metric on seed-paired means."""
    m = lambda arm, part, k: float(np.mean([models[f'{arm}_s{s}']['mean'][part][k] for s in seeds]))
    return {part: {k: (m('C2MS', part, k) - m('C2', part, k)) - (m('C0MS', part, k) - m('C0', part, k)) for k in ('PSNR', 'SAM', 'ERGAS')} for part in PARTS}


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
            if c: ax[1].plot([r['epoch'] for r in c], [r['coarse_val_psnr'] for r in c], **kw)
    ax[0].set(xlabel='epoch', ylabel='PSNR (dB)', title='QB validation, original scale (selection)', ylim=(37, 41))
    ax[1].set(xlabel='epoch', ylabel='PSNR (dB)', title='QB validation, 2x coarser pairs (logged only)')
    for x in ax: x.legend(fontsize=8); x.grid(alpha=.3)
    fig.suptitle('Learning curves, seed 42 bold and seeds 43-44 light (C0, C2 from 43; C2MS from 53; C0MS from 56)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=130); plt.close(fig)
    x = np.arange(len(PARTS)); w = .26; three = '+'.join(map(str, SEEDS))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for j, arm in enumerate(('C0MS', 'C2', 'C2MS')):
        key = f'{arm}:C0@{three}'
        v = [judgements[key]['delta'][t]['PSNR'] for t in PARTS]; per = [[judgements[f'{arm}:C0@{s}']['delta'][t]['PSNR'] for s in SEEDS] for t in PARTS]
        ax.bar(x + (j - 1) * w, v, w, color=COLORS[arm], label=f'{arm} - C0')
        for xi, d in zip(x + (j - 1) * w, per): ax.scatter([xi] * len(d), d, color='black', s=8, zorder=3)
        for xi, vi in zip(x + (j - 1) * w, v): ax.text(xi, vi + (0.05 if vi >= 0 else -0.15), f'{vi:+.2f}', ha='center', fontsize=8)
    ax.axhline(0, color='black', lw=.8); ax.set_xticks(x, PARTS); ax.set_ylabel('PSNR difference vs C0, same seed (dB)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
    ax.set_title('QB 2x2: mixing alone (C0MS), W2 alone (C2), both (C2MS); bars = 3-seed mean, dots = seeds', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(CFG['runtime']), help='56 runtime with runs/QB_C0MS_s<seed>/ (until the weights are archived)')
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
    scenes = {}
    for name, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename)); assert len(found) == 1, filename
        digest = V.sha(found[0])
        if digest != CFG['data_sha256']['QB'][name]: raise ValueError('data sha mismatch ' + name)
        with h5py.File(found[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if name == 'RR' else ('pan', 'lms', 'ms')
            n = len(f['pan']) if a.limit is None else a.limit
            scenes[name] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(n)]
        scenes[name + '_sha256'] = digest
    degrade = V.protocol.WaldDegradation('QB', a.device)
    observation = V.protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    nets, models = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            path = checkpoint(arm, s, a.runtime); ck = torch.load(path, map_location='cpu', weights_only=False)
            if arm == 'C0MS' and (ck.get('sensor'), ck.get('arm'), ck.get('seed')) != ('QB', 'C0MS', s): raise ValueError(f'checkpoint identity {path}')
            net = V.M53.M48.Model(BUILD[arm], s); net.load_state_dict(ck['model'], strict=True); net = net.to(a.device).eval()
            rows = [{'index': i, **V.protocol.scene_metrics(net, rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
            mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
            models[f'{arm}_s{s}'] = dict(checkpoint=str(path), checkpoint_sha256=V.sha(path), best_epoch=ck.get('best_epoch'), mean=mean, rows=rows)
            print(arm, s, 'RR %.4f/%.4f/%.4f RR2 %.4f/%.4f FR_WALD %.4f/%.4f OBS %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'], mean['RR2']['PSNR'],
                  mean['RR2']['SAM'], mean['FR_WALD']['PSNR'], mean['FR_WALD']['SAM'], mean['FR']['OBS_RMSE_DN']), flush=True)
            if s == 42: nets[arm] = net
            else: del net
    judgements = {}
    for cand, ctrl in PAIRS:
        for seeds in (SEEDS,) + tuple((s,) for s in SEEDS):
            name = f'{cand}:{ctrl}@' + '+'.join(map(str, seeds))
            j = V.judge([models[f'{cand}_s{s}']['mean'] for s in seeds], [models[f'{ctrl}_s{s}']['mean'] for s in seeds]); j['seeds'] = list(seeds)
            j['scenes_improved'] = V.scene_wins([models[f'{cand}_s{s}']['rows'] for s in seeds], [models[f'{ctrl}_s{s}']['rows'] for s in seeds]); judgements[name] = j
            print(name, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d' % (
                j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
                j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR']), flush=True)
    inter = {'+'.join(map(str, seeds)): interaction(models, seeds) for seeds in (SEEDS,) + tuple((s,) for s in SEEDS)}
    three = '+'.join(map(str, SEEDS))
    print('interaction (C2MS-C2)-(C0MS-C0) PSNR', {part: round(v['PSNR'], 4) for part, v in inter[three].items()}, flush=True)
    prefixed = {'primary_C0MS_vs_C0': judgements[f'C0MS:C0@{three}']['label'], 'C2_vs_C0': judgements[f'C2:C0@{three}']['label'],
                'C2MS_vs_C2': judgements[f'C2MS:C2@{three}']['label'], 'C2MS_vs_C0MS': judgements[f'C2MS:C0MS@{three}']['label']}
    consistency = {}
    if a.limit is None:
        ref = json.loads((E53 / 'results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models']
        for arm in ('C0', 'C2', 'C2MS'):
            for s in SEEDS:
                r = ref[f'{arm}_s{s}']['mean']; m = models[f'{arm}_s{s}']['mean']
                consistency[f'{arm}_s{s}:vs_53'] = max(abs(m[part][k] - r[part][k]) for part in PARTS for k in ('PSNR', 'SAM', 'ERGAS'))
        for s in SEEDS:
            rec = a.runtime / f'runs/QB_C0MS_s{s}/evaluation/RR.json'
            if not rec.exists(): rec = HERE / f'results/C0MS_QB_s{s}/evaluation/RR.json'
            if rec.exists():
                r = json.loads(rec.read_text(encoding='utf-8'))['mean']
                consistency[f'C0MS_s{s}:vs_controller'] = max(abs(models[f'C0MS_s{s}']['mean']['RR'][k] - r[k]) for k in ('PSNR', 'SAM', 'ERGAS'))
    print('pre-fixed', prefixed, 'max consistency diff', max(consistency.values(), default=None), flush=True)
    V.result_images.render(nets['C0'], nets['C0MS'], a.data, out / 'images/C0MS_vs_original', names=('원본', 'C0MS'), seed=42, device=a.device,
                           scenes=tuple(i for i in CFG['scenes'] if a.limit is None or i < a.limit), labels=('원본 SSA-MRN', 'C0MS (원본 + 축척 혼합)'))
    graphs(models, judgements, a.runtime, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3 (common/evaluation_protocol, QB)', noise=V.NOISE, tolerances=V.TOL, seeds=list(SEEDS), arms=list(ARMS), pairs=PAIRS,
        scenes_per_test=a.limit or 20, prefixed_judgement=prefixed, judgements=judgements, interaction=inter, consistency=consistency, models=models,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=V.sha(__file__),
                         evaluate53_sha256=V.sha(E53 / 'evaluate.py'), protocol_sha256=V.sha(V.ROOT / 'common/evaluation_protocol.py'),
                         result_images_sha256=V.sha(V.ROOT / 'common/result_images.py'), rr_sha256=scenes['RR_sha256'], fr_sha256=scenes['FR_sha256'])),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
