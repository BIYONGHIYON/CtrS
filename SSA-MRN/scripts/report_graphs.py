"""Compact measured epoch records and plot training/validation plus test baseline."""
import json
from pathlib import Path


def compact_history(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        note = 'Available records only; missing epochs are not reconstructed.'
    else:
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        rows = obj.get('epochs', []) if isinstance(obj, dict) else obj
        note = obj.get('note', '') if isinstance(obj, dict) else ''
    keys = ('epoch', 'train_mse', 'model_mse', 'model_psnr_db', 'model_sam_deg',
            'seconds', 'train_seconds', 'validation_seconds', 'loader_wait_seconds',
            'train_patches_per_second', 'peak_allocated_mb', 'peak_reserved_mb')
    # A best-resume may repeat epochs; preserve the final observed record per epoch.
    latest = {row['epoch']: {k: row[k] for k in keys if k in row} for row in rows}
    return {'note': note, 'epochs': [latest[k] for k in sorted(latest)]}


def plot_report(curves, metrics, output, previous=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = curves['epochs']
    if not rows:
        raise ValueError('No measured epoch records: a training graph is required')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, label in zip(axes, ('train_mse', 'model_psnr_db', 'model_sam_deg'),
                             ('Training MSE', 'Validation PSNR (dB)', 'Validation SAM (deg)')):
        ax.plot([x['epoch'] for x in rows], [x.get(key, float('nan')) for x in rows])
        ax.set(title=label, xlabel='Epoch')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / 'learning.png', dpi=140)
    plt.close(fig)
    mean = metrics['scene_mean']
    base = 'interp23' if 'interp23_mse' in mean else 'bicubic'
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, label in zip(axes, ('mse', 'psnr_db', 'sam_deg'), ('Test MSE', 'Test PSNR (dB)', 'Test SAM (deg)')):
        names, values = [base, 'model'], [mean[base+'_'+key], mean['model_'+key]]
        if previous is not None:
            names.insert(0, 'previous*')
            values.insert(0, previous['scene_mean']['model_'+key])
        ax.bar(names, values)
        ax.set_title(label)
    fig.suptitle('Scene means; * previous protocol must be checked separately')
    fig.tight_layout()
    fig.savefig(output / 'test_metrics.png', dpi=140)
    plt.close(fig)
