"""Experiment 57 evaluation (no training): seed 42 screening of C2MS training conditions on QB.

Candidates from this experiment (online best.pt and EMA best_ema.pt of each run): C2MS (reproduction / + EMA), C2MS_SAM
(+ spectral loss / + spectral loss + EMA). The 300-epoch run C2MS_L300 was stopped at epoch 4 by the user's decision and is not
evaluated. Control: 53 C2MS seed 42. Metrics: common/evaluation_protocol.scene_metrics on all 20 RR and FR QB test scenes,
judged with the 53 evaluate.py v2.3 functions (single seed, noise threshold 0.06 dB). Also the candidates vs the 57 C2MS online
weights (same-run lineage), the online RR means against the controller records, QB images (53 C2MS vs C2MS_SAM, seed 42) and graphs.
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
SEED = 42
CANDIDATES = {'C2MS': ('C2MS', 'best.pt'), 'C2MS_EMA': ('C2MS', 'best_ema.pt'), 'C2MS_SAM': ('C2MS_SAM', 'best.pt'), 'C2MS_SAM_EMA': ('C2MS_SAM', 'best_ema.pt')}
COLORS = {'C2MS': '#8C8C8C', 'C2MS_EMA': '#BDBDBD', 'C2MS_SAM': '#C44E52', 'C2MS_SAM_EMA': '#E39A9C', 'control': '#000000'}
PARTS = ('RR', 'RR2', 'FR_WALD')
GOAL = {'name': 'U2Net + PAKAN (2026, paper value)', 'RR': {'PSNR': 38.520, 'SAM': 4.439, 'ERGAS': 3.698}}


def checkpoint(name, runtime):
    arm, fname = CANDIDATES[name]
    archived = HERE / f'checkpoints/{arm}_QB_s{SEED}/{fname}'
    return archived if archived.exists() else runtime / f'runs/QB_{arm}_s{SEED}/checkpoints/{fname}'


def history(arm, runtime):
    archived = HERE / f'results/{arm}_QB_s{SEED}/history.json'
    return archived if archived.exists() else runtime / f'runs/QB_{arm}_s{SEED}/history.json'


def graphs(models, judgements, runtime, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    h53 = [r for r in json.loads(V.history('C2MS', SEED).read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
    ax[0].plot([r['epoch'] for r in h53], [r['val_psnr'] for r in h53], color=COLORS['control'], lw=1.2, label='53 C2MS (control)')
    for arm in ('C2MS', 'C2MS_SAM'):
        p = history(arm, runtime)
        if not p.exists(): continue
        h = [r for r in json.loads(p.read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
        ax[0].plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], color=COLORS[arm], lw=1.2, label=f'57 {arm} online')
        e = [r for r in h if r.get('ema_val_psnr') is not None]
        ax[0].plot([r['epoch'] for r in e], [r['ema_val_psnr'] for r in e], color=COLORS[arm + '_EMA'], lw=1.0, ls='--', label=f'57 {arm} EMA')
    ax[0].set(xlabel='epoch', ylabel='PSNR (dB)', title='QB validation, original scale (seed 42)', ylim=(39.5, 40.4)); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    names = list(CANDIDATES); x = np.arange(len(PARTS)); w = .2
    for j, n in enumerate(names):
        v = [judgements[f'{n}:control']['delta'][t]['PSNR'] for t in PARTS]
        ax[1].bar(x + (j - 1.5) * w, v, w, color=COLORS[n], label=n)
        for xi, vi in zip(x + (j - 1.5) * w, v): ax[1].text(xi, vi + (0.01 if vi >= 0 else -0.03), f'{vi:+.3f}', ha='center', fontsize=7)
    ax[1].axhline(0, color='black', lw=.8); ax[1].axhline(0.06, color='#888888', lw=.8, ls=':'); ax[1].set_xticks(x, PARTS); ax[1].legend(fontsize=7); ax[1].grid(axis='y', alpha=.3)
    ax[1].set(ylabel='PSNR difference vs 53 C2MS (dB)', title='QB test 20 scenes, seed 42 (dotted: single-seed noise threshold)')
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(CFG['runtime']))
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='first N scenes only (code check)')
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
    paths = {'control': V.ckpt('C2MS', SEED), **{n: checkpoint(n, a.runtime) for n in CANDIDATES}}
    nets, models = {}, {}
    for name, path in paths.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if name != 'control' and (ck.get('sensor'), ck.get('arm'), ck.get('seed')) != ('QB', CANDIDATES[name][0], SEED): raise ValueError(f'checkpoint identity {path}')
        net = V.M53.M48.Model('C2', SEED); net.load_state_dict(ck['model'], strict=True); net = net.to(a.device).eval(); nets[name] = net
        rows = [{'index': i, **V.protocol.scene_metrics(net, rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
        mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
        models[name] = dict(checkpoint=str(path), checkpoint_sha256=V.sha(path), best_epoch=ck.get('best_epoch'), mean=mean, rows=rows)
        print(name, 'epoch', ck.get('best_epoch'), 'RR %.4f/%.4f/%.4f RR2 %.4f/%.4f FR_WALD %.4f/%.4f OBS %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'],
              mean['RR2']['PSNR'], mean['RR2']['SAM'], mean['FR_WALD']['PSNR'], mean['FR_WALD']['SAM'], mean['FR']['OBS_RMSE_DN']), flush=True)
    judgements = {}
    pairs = [(n, 'control') for n in CANDIDATES] + [(n, 'C2MS') for n in ('C2MS_EMA', 'C2MS_SAM', 'C2MS_SAM_EMA')]
    for cand, ctrl in pairs:
        j = V.judge([models[cand]['mean']], [models[ctrl]['mean']]); j['scenes_improved'] = V.scene_wins([models[cand]['rows']], [models[ctrl]['rows']])
        judgements[f'{cand}:{ctrl}'] = j
        print(f'{cand}:{ctrl}', j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d' % (
            j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
            j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR']), flush=True)
    gap = {n: {k: models[n]['mean']['RR'][k] - GOAL['RR'][k] for k in ('PSNR', 'SAM', 'ERGAS')} for n in list(CANDIDATES) + ['control']}
    consistency = {}
    if a.limit is None:
        for arm in ('C2MS', 'C2MS_SAM'):
            rec = a.runtime / f'runs/QB_{arm}_s{SEED}/evaluation/RR.json'
            if not rec.exists(): rec = HERE / f'results/{arm}_QB_s{SEED}/evaluation/RR.json'
            if rec.exists():
                r = json.loads(rec.read_text(encoding='utf-8'))['mean']; consistency[f'{arm}:vs_controller'] = max(abs(models[arm]['mean']['RR'][k] - r[k]) for k in ('PSNR', 'SAM', 'ERGAS'))
        ref = json.loads((E53 / 'results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models'][f'C2MS_s{SEED}']['mean']
        consistency['control:vs_53'] = max(abs(models['control']['mean'][part][k] - ref[part][k]) for part in PARTS for k in ('PSNR', 'SAM', 'ERGAS'))
    print('gap to goal (RR)', {n: {k: round(v, 3) for k, v in g.items()} for n, g in gap.items()}, 'max consistency diff', max(consistency.values(), default=None), flush=True)
    V.result_images.render(nets['control'], nets['C2MS_SAM'], a.data, out / 'images/C2MS_SAM_vs_C2MS', names=('C2MS', 'C2MS+분광'), seed=SEED, device=a.device,
                           scenes=tuple(i for i in CFG['scenes'] if a.limit is None or i < a.limit), labels=('현재 기준 C2MS', 'C2MS + 분광 손실'))
    graphs(models, judgements, a.runtime, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3 (common/evaluation_protocol, QB), single seed', noise=V.NOISE, tolerances=V.TOL, seed=SEED, candidates=CANDIDATES,
        stopped_runs={'C2MS_L300': 'stopped at epoch 4 on 2026-10-24 by the user decision (300 epochs deferred until the better of C2MS / C2MS_SAM is known); not evaluated'},
        scenes_per_test=a.limit or 20, judgements=judgements, goal=GOAL, gap_to_goal=gap, consistency=consistency, models=models,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=V.sha(__file__),
                         evaluate53_sha256=V.sha(E53 / 'evaluate.py'), protocol_sha256=V.sha(V.ROOT / 'common/evaluation_protocol.py'),
                         result_images_sha256=V.sha(V.ROOT / 'common/result_images.py'), rr_sha256=scenes['RR_sha256'], fr_sha256=scenes['FR_sha256'])),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
