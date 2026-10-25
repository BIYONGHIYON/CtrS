"""61 phase 1 screen report (no training): v2.3 judgements, SAM goal rows, graphs and result images.

Reads the server evaluation records results/<run>/evaluation_complete.json (control C2MS and C2MS + selected head on the
same 20 QB RR/FR scenes, seed 42) and results/<run>/history.json. Judgement: docs/judgement_criteria.md v2.3, single seed
(noise threshold 0.06 dB), candidate vs the same-seed C2MS. The SAM goal (config judgement.sam_goal) is reported separately.
Images: common/result_images.py with the validation-selected main candidate (H2 at the H3 lambda) against C2MS, CPU.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
from common import result_images  # noqa: E402

RUNS = ('QB_H1_l0_s42', 'QB_H2_l0.003_s42', 'QB_H2_l0.01_s42', 'QB_H2_l0.03_s42', 'QB_H3_l0.003_s42')
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


def graphs(hist, judgements, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for run in RUNS:
        h = hist[run]; e = [r['epoch'] for r in h]
        ax[0].plot(e, [r['val_psnr'] - h[0]['val_psnr'] for r in h], label=run.replace('QB_', '').replace('_s42', ''))
        ax[1].plot(e, [r['val_sam'] - h[0]['val_sam'] for r in h], label=run.replace('QB_', '').replace('_s42', ''))
    ax[0].axhline(-0.02, ls=':', c='gray'); ax[0].set_title('validation PSNR - C2MS (dB); dotted: selection floor'); ax[0].set_xlabel('epoch')
    ax[1].set_title('validation SAM - C2MS (deg)'); ax[1].set_xlabel('epoch'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); x = np.arange(len(RUNS)); w = 0.27
    for i, part in enumerate(('RR', 'RR2', 'FR_WALD')):
        ax[0].bar(x + (i - 1) * w, [judgements[r]['delta'][part]['PSNR'] for r in RUNS], w, label=part)
        ax[1].bar(x + (i - 1) * w, [judgements[r]['delta'][part]['SAM'] for r in RUNS], w, label=part)
    ax[0].axhline(NOISE, ls=':', c='gray'); ax[1].axhline(-0.15, ls=':', c='gray')
    for a, t in zip(ax, ('PSNR - C2MS (dB); dotted: single-seed noise threshold', 'SAM - C2MS (deg); dotted: first SAM goal')):
        a.set_xticks(x, [r.replace('QB_', '').replace('_s42', '') for r in RUNS], fontsize=8); a.set_title(t); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=120); plt.close(fig)


class Full(torch.nn.Module):
    def __init__(self, c2ms, head):
        super().__init__(); self.c2ms, self.head = c2ms, head

    def forward(self, pan, lms, ms):
        y = self.c2ms(pan, lms, ms)
        return y if self.head is None else self.head(y, pan, lms)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, default=ROOT / 'data'); p.add_argument('--skip-images', action='store_true'); a = p.parse_args()
    out = HERE / 'results'; recs = {r: json.loads((out / r / 'evaluation_complete.json').read_text(encoding='utf-8')) for r in RUNS}
    hist = {r: json.loads((out / r / 'history.json').read_text(encoding='utf-8')) for r in RUNS}
    judgements = {r: dict(selected_epoch=recs[r]['selected_epoch'], **judge(recs[r]['targets']['QB'])) for r in RUNS}
    lam = json.loads((out / 'QB_H3_lambda_s42.json').read_text(encoding='utf-8'))['lam']; main_run = f'QB_H2_l{lam:g}_s42'
    result = dict(criteria='docs/judgement_criteria.md v2.3 (single seed 42, noise 0.06 dB)', control='C2MS QB seed 42 (53 best.pt)',
                  main_candidate=main_run, main_candidate_rule=CFG['phase1']['h3_lambda_rule'], judgements=judgements)
    (out / 'judgement_v2_3.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    graphs(hist, judgements, out)
    for r in RUNS:
        j = judgements[r]; print(r, j['label'], {k: round(v, 4) for k, v in j['delta']['RR'].items() if k in ('PSNR', 'SAM', 'ERGAS')}, j['guards'], round(j['obs_ratio'], 4), j['scene_wins']['RR'])
    if not a.skip_images:
        torch.set_num_threads(4)
        c2ms = M.C2MS('QB', 42); c2ms.load_state_dict(torch.load(ROOT / CFG['phase1']['controls']['QB'].replace('<seed>', '42'), map_location='cpu', weights_only=False)['model'], strict=True)
        _, arm, l, _ = main_run.split('_'); head = M.Head(arm, 'QB', 42)
        head.load_state_dict(torch.load(HERE / 'checkpoints' / main_run / 'best.pt', map_location='cpu', weights_only=False)['model'], strict=True)
        result_images.render(Full(c2ms, None).eval(), Full(c2ms, head).eval(), a.data, out / 'images' / f'{main_run}_vs_C2MS', names=('C2MS', f'H2 λ{lam:g}'), seed=42, device='cpu')


if __name__ == '__main__':
    main()
