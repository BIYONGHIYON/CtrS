"""Summarize completed validation runs and plot pilot learning curves."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    curves = []
    for run in args.runs:
        cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8-sig'))
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if history[-1]['epoch'] != cfg['epochs']:
            raise RuntimeError('Run has not reached requested epochs: ' + str(run))
        if any(row.get('limited_scenes') for row in history):
            raise RuntimeError('Limited-scene benchmark cannot enter full-scene pilot comparison')
        best = min(history, key=lambda row: row['model_mse'])
        metrics = json.loads((run / 'best_validation_metrics.json').read_text(encoding='utf-8-sig'))
        if len(metrics['scenes']) != 45:
            raise RuntimeError('Pilot requires all 45 validation scenes')
        entries.append({'label': cfg['pilot_label'], 'run': str(run.resolve()),
            'best_epoch': best['epoch'], 'epochs': cfg['epochs'], 'seed': cfg['seed'],
            'model_type': cfg['model_type'], 'options': cfg.get('pilot_options', {}),
            'initialization_sha256': cfg.get('initialization', {}).get('sha256'),
            'seconds': sum(row['seconds'] for row in history),
            'checkpoint_sha256': hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(),
            'metrics': metrics['scene_mean']})
        curves.append((cfg['pilot_label'], history))
    # Distinguish scratch and warm starts in reports; never rank them as one pool.
    groups = {'scratch': [], 'warm': []}
    for item in entries:
        groups['warm' if item['initialization_sha256'] else 'scratch'].append(item)
    (args.output / 'summary.json').write_text(json.dumps({'split': 'validation', 'groups': groups,
        'note': 'Screening only; short-training rankings may change after full training.'}, indent=2), encoding='utf-8')
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, rows in curves:
        for axis, key in zip(axes, ('train_mse', 'model_psnr_db', 'model_sam_deg')):
            axis.plot([row['epoch'] for row in rows], [row[key] for row in rows], label=label)
            axis.set_xlabel('Epoch')
            axis.set_title(key)
            axis.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / 'learning.png', dpi=160)
    plt.close(figure)
    lines = ['# 예비 실험 검증 결과', '', 'validation45만 사용한 후보 선별 결과입니다. 정식 test 성능은 아닙니다.', '']
    for label, items in groups.items():
        if not items:
            continue
        lines += ['## ' + label, '', '| 후보 | best epoch | PSNR dB | SAM ° | gradient RMSE | 시간 분 |',
                  '|---|---:|---:|---:|---:|---:|']
        for item in items:
            m = item['metrics']
            lines.append(f"| {item['label']} | {item['best_epoch']} | {m['model_psnr_db']:.4f} | {m['model_sam_deg']:.4f} | {m['model_gradient_rmse']:.6f} | {item['seconds']/60:.1f} |")
        lines.append('')
    lines += ['![예비 학습 곡선](learning.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
