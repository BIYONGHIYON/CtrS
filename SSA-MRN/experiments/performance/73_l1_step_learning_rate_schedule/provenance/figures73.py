"""Draw the 73 report graphs from the archived results only (no torch, no test data):
results/comparison.png       test PSNR/SAM deltas of both arms and the 63 L1 model against the 53 C2MS (RR, RR2, FR_WALD)
results/validation_curves.png validation PSNR/SAM per epoch of 53, 63 and both arms, zoomed to the converged range
python SSA-MRN/experiments/performance/73_l1_step_learning_rate_schedule/provenance/figures73.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
PERF = HERE.parent
load = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
ARMS = {'C2MS_L1S': 'L1S (4e-4 step, 360)', 'C2MS_L1C': 'L1C (1e-4 cosine, 360)'}
records = {arm: load(HERE / f'results/{arm}_QB_s42/evaluation/QB_v2_3.json') for arm in ARMS}
means = {arm: r['means'][arm] for arm, r in records.items()}
means['C2MS_L1_63'] = records['C2MS_L1C']['means']['C2MS_L1_63']
control = records['C2MS_L1C']['means']['C2MS']
labels = {**ARMS, 'C2MS_L1_63': '63 L1 (1e-4 cosine, 150)'}
colors = {'C2MS_L1S': '#d1495b', 'C2MS_L1C': '#00798c', 'C2MS_L1_63': '#8d99ae', 'C2MS_53': '#333333'}

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
parts = ('RR', 'RR2', 'FR_WALD')
for ax, key, unit in zip(axes, ('PSNR', 'SAM'), ('dB', 'degrees')):
    width = .26
    for j, name in enumerate(('C2MS_L1_63', 'C2MS_L1C', 'C2MS_L1S')):
        values = [means[name][p][key] - control[p][key] for p in parts]
        ax.bar([i + (j - 1) * width for i in range(len(parts))], values, width, label=labels[name], color=colors[name])
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(range(len(parts)), parts)
    ax.set(ylabel=f'{key} - 53 C2MS ({unit})', title=f'QB test {key} against 53 C2MS, seed 42')
    ax.grid(axis='y', alpha=.3)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'results/comparison.png', dpi=140)
plt.close(fig)

histories = {'C2MS_53': load(PERF / '53_scale_mixed_training/results/C2MS_QB_s42/history.json'),
             'C2MS_L1_63': load(PERF / '63_l1_reconstruction_loss/results/C2MS_L1_QB_s42/history.json'),
             **{arm: load(HERE / f'results/{arm}_QB_s42/history.json') for arm in ARMS}}
names = {'C2MS_53': '53 C2MS (MSE, 1e-4 cosine, 150)', **labels}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, key, limits in zip(axes, ('val_psnr', 'val_sam'), ((39.8, 40.6), (4.10, 4.40))):
    for name, history in histories.items():
        rows = [x for x in history if x.get(key) is not None and x['epoch'] > 0]
        ax.plot([x['epoch'] for x in rows], [x[key] for x in rows], lw=1.2, label=names[name], color=colors[name])
    ax.set(xlabel='Epoch', ylabel=key.replace('val_', 'Validation ').upper().replace('VALIDATION', 'Validation'), ylim=limits)
    ax.grid(alpha=.3)
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('QB validation, seed 42 (weights selected by validation MSE)')
fig.tight_layout()
fig.savefig(HERE / 'results/validation_curves.png', dpi=140)
plt.close(fig)
