"""62 screen report (no training): gates, v2.3 judgements, SAM goal rows, graphs and result images.

Reads the server records results/<run>/{history,training_complete,evaluation_complete}.json (control C2MS and C2MS +
selected head on the same 20 QB RR/FR scenes, seed 42). Gates: config.json gates, on the selected epoch's validation SAM
and the training-probe SAM. Judgement: docs/judgement_criteria.md v2.3, single seed (noise 0.06 dB), vs the same-seed C2MS.
Images: common/result_images.py with the arm whose selected epoch has the lowest validation SAM, against C2MS, on CPU.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
from common import result_images  # noqa: E402

RUNS = ('QB_F0_s42', 'QB_F1_s42', 'QB_F2_s42', 'QB_F3_s42')
NOISE = 0.06
TOL = {'RR2': (-0.05, 0.06), 'FR_WALD': (-0.07, 0.04), 'FR_OBS': 1.10}
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def judge(rec):
    c, b = rec['candidate']['mean'], rec['C2MS']['mean']
    d = {part: {k: c[part][k] - b[part][k] for k in b[part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
    obs = c['FR']['OBS_RMSE_DN'] / b['FR']['OBS_RMSE_DN']
    rr = {'PSNR': d['RR']['PSNR'] > 0, 'SAM': d['RR']['SAM'] < 0, 'ERGAS': d['RR']['ERGAS'] < 0}
    guards = {k: d[k]['PSNR'] >= TOL[k][0] and d[k]['SAM'] <= TOL[k][1] for k in ('RR2', 'FR_WALD')}; guards['FR_OBS'] = obs <= TOL['FR_OBS']
    above = d['RR']['PSNR'] >= NOISE
    label = '개선' if all(rr.values()) and above and all(guards.values()) else '부분 개선' if any(rr.values()) else '개선 없음'
    wins = {part: dict(PSNR=int(sum(x[part]['PSNR'] > y[part]['PSNR'] for x, y in zip(rec['candidate']['scenes'], rec['C2MS']['scenes']))),
                       SAM=int(sum(x[part]['SAM'] < y[part]['SAM'] for x, y in zip(rec['candidate']['scenes'], rec['C2MS']['scenes'])))) for part in ('RR', 'RR2', 'FR_WALD')}
    goal = dict(first=bool(d['RR']['SAM'] <= -0.15 and d['RR']['PSNR'] >= -0.04), stretch=bool(d['RR']['SAM'] <= -0.28 and d['RR']['PSNR'] >= -0.04))
    return dict(delta=d, obs_ratio=obs, noise_threshold=NOISE, above_noise=above, rr_components_improved=rr, guards=guards, label=label,
                scene_wins=wins, sam_goal=goal, control_mean=b, candidate_mean=c)


def gates(done, hist):
    s, z = done['selected'], done['epoch0']
    val = z['val_sam'] - s['val_sam']; probe = z['train_probe_sam'] - s['train_probe_sam']
    probe_best = z['train_probe_sam'] - min(h['train_probe_sam'] for h in hist)
    return dict(selected_epoch=done['selected_epoch'], parameters=done['parameters'], val_sam_decrease=val, val_psnr_change=s['val_psnr'] - z['val_psnr'],
                train_probe_sam_decrease=probe, train_probe_sam_decrease_best_epoch=probe_best, train_probe_psnr_change=s['train_probe_psnr'] - z['train_probe_psnr'],
                expand=bool(val >= 0.10 and s['val_psnr'] >= z['val_psnr'] - 0.02))


def graphs(hist, judgements, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); short = lambda r: r.replace('QB_', '').replace('_s42', '')
    for run in RUNS:
        h = hist[run]; e = [r['epoch'] for r in h]
        ax[0].plot(e, [r['val_psnr'] - h[0]['val_psnr'] for r in h], label=short(run))
        ax[1].plot(e, [r['val_sam'] - h[0]['val_sam'] for r in h], label=short(run))
        ax[2].plot(e, [r['train_probe_sam'] - h[0]['train_probe_sam'] for r in h], label=short(run))
    ax[0].set_title('validation PSNR - C2MS (dB)'); ax[1].set_title('validation SAM - C2MS (deg)'); ax[2].set_title('training-probe SAM - C2MS (deg)')
    for a in ax[1:]: a.axhline(-0.10, ls=':', c='gray')
    for a in ax: a.set_xlabel('epoch'); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); x = np.arange(len(RUNS)); w = 0.27
    for i, part in enumerate(('RR', 'RR2', 'FR_WALD')):
        ax[0].bar(x + (i - 1) * w, [judgements[r]['delta'][part]['PSNR'] for r in RUNS], w, label=part)
        ax[1].bar(x + (i - 1) * w, [judgements[r]['delta'][part]['SAM'] for r in RUNS], w, label=part)
    ax[0].axhline(NOISE, ls=':', c='gray'); ax[1].axhline(-0.15, ls=':', c='gray')
    for a, t in zip(ax, ('PSNR - C2MS (dB); dotted: single-seed noise threshold', 'SAM - C2MS (deg); dotted: first SAM goal')):
        a.set_xticks(x, [short(r) for r in RUNS]); a.set_title(t); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, default=ROOT / 'data'); p.add_argument('--skip-images', action='store_true'); a = p.parse_args()
    out = HERE / 'results'; load = lambda r, f: json.loads((out / r / f).read_text(encoding='utf-8'))
    recs = {r: load(r, 'evaluation_complete.json') for r in RUNS}; hist = {r: load(r, 'history.json') for r in RUNS}; done = {r: load(r, 'training_complete.json') for r in RUNS}
    g = {r: gates(done[r], hist[r]) for r in RUNS}
    if any(x['expand'] for x in g.values()): decision = 'expand'
    elif all(x['train_probe_sam_decrease'] < 0.10 for x in g.values()): decision = 'stop_information_limit'
    else: decision = 'generalization_limit'
    judgements = {r: dict(selected_epoch=recs[r]['selected_epoch'], **judge(recs[r]['targets']['QB'])) for r in RUNS}
    main_run = min(RUNS, key=lambda r: done[r]['selected']['val_sam'])
    result = dict(criteria='docs/judgement_criteria.md v2.3 (single seed 42, noise 0.06 dB)', control='C2MS QB seed 42 (53 best.pt)',
                  gates=g, gate_decision=decision, gate_rule=CFG['gates'], image_candidate=main_run,
                  image_candidate_rule='arm whose selected epoch has the lowest validation SAM', judgements=judgements)
    (out / 'judgement_v2_3.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    graphs(hist, judgements, out)
    print('gate decision', decision, '| image candidate', main_run)
    for r in RUNS:
        j = judgements[r]; print(r, j['label'], {k: round(v, 4) for k, v in j['delta']['RR'].items() if k in ('PSNR', 'SAM', 'ERGAS')}, j['guards'], round(j['obs_ratio'], 4), j['scene_wins']['RR'],
                                  {k: round(v, 4) if isinstance(v, float) else v for k, v in g[r].items()})
    if not a.skip_images:
        torch.set_num_threads(4); arm = main_run.split('_')[1]
        c2ms = M.C2MSWithFeatures('QB', 42)
        c2ms.net.load_state_dict(torch.load(ROOT / CFG['phase1']['controls']['QB'].replace('<seed>', '42'), map_location='cpu', weights_only=False)['model'], strict=True)
        head = M.Head(arm, 'QB', 42); head.load_state_dict(torch.load(HERE / 'checkpoints' / main_run / 'best.pt', map_location='cpu', weights_only=False)['model'], strict=True)
        result_images.render(M.Full(c2ms.eval()).eval(), M.Full(c2ms.eval(), head.eval()).eval(), a.data, out / 'images' / f'{main_run}_vs_C2MS', names=('C2MS', arm), seed=42, device='cpu')


if __name__ == '__main__':
    main()
