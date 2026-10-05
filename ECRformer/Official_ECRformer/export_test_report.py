"""Evaluate a checkpoint and export a compact, reproducible report (no training)."""
import argparse
import contextlib
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import test as evaluation
from train import get_subset_paths
from util.checkpoint import load_checkpoint_file


METRICS = ('RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hash_chunks(stream)


def hash_chunks(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')


def metric_values(prediction, target):
    result = evaluation.compute_metric(prediction.float(), target.float(), size_average=False)
    result = {key: float(result[key][0]) for key in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f'Non-finite evaluation metric: {result}')
    return result


def choose_examples(paths, seed):
    groups = {}
    for index, item in enumerate(paths):
        groups.setdefault(Path(item['S1']).parent.name, []).append(index)
    rng = random.Random(seed)
    rois = sorted(groups)
    rng.shuffle(rois)
    chosen = [rng.choice(groups[roi]) for roi in rois[:5]]
    if len(chosen) < min(5, len(paths)):
        remaining = [index for index in range(len(paths)) if index not in chosen]
        chosen += rng.sample(remaining, min(5, len(paths)) - len(chosen))
    return chosen, len(groups)


def plot_case(output, number, case):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    axes[0].imshow(np.clip(case['sar'][0], 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('SAR (VV)')
    for ax, key, title in zip(axes[1:], ['cloudy', 'best', 'last', 'target'],
                              ['Cloudy input', 'Best (epoch 8)', 'Last (epoch 18)', 'Target']):
        ax.imshow(evaluation.to_rgb_image(case[key], [3, 2, 1], brightness=3.0))
        ax.set_title(title)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(case['sample_id'], fontsize=10)
    fig.tight_layout()
    fig.savefig(output / f'sample_{number:02d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_curves(output, diagnostics):
    epochs = [row for row in diagnostics if row['event'] == 'EPOCH_SUMMARY']
    validation = [row for row in diagnostics if row['event'] == 'VALIDATION_SUMMARY']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot([r['epoch'] for r in epochs], [r['train_mae_mean'] for r in epochs], label='Train MAE')
    axes[0, 0].plot([r['epoch'] for r in validation], [r['valid_MAE'] for r in validation], label='Validation MAE')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('MAE (log scale)')
    axes[0, 0].legend()
    for ax, key, title in [(axes[0, 1], 'valid_PSNR', 'Validation PSNR (dB)'),
                            (axes[1, 0], 'valid_SAM', 'Validation SAM (degrees)'),
                            (axes[1, 1], 'valid_SSIM', 'Validation SSIM')]:
        ax.plot([r['epoch'] for r in validation], [r[key] for r in validation], marker='.')
        ax.set_title(title)
    for ax in axes.flat:
        ax.axvline(8, linestyle='--', color='green', alpha=.6)
        ax.axvline(16, linestyle='--', color='red', alpha=.6)
        ax.set_xlabel('Epoch (zero-based)')
        ax.grid(alpha=.2)
    fig.suptitle('Recorded resume segment only | green: best; red: divergence')
    fig.tight_layout()
    fig.savefig(output / 'learning.png', dpi=130)
    plt.close(fig)


def plot_metrics(output, summary):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ['PSNR', 'SAM', 'SSIM'],
                              ['PSNR (dB), higher is better', 'SAM (deg), lower is better', 'SSIM, higher is better']):
        values = [summary['cloudy_baseline'][key], summary['best_test'][key]]
        ax.bar(['Cloudy input', 'Best model'], values, color=['#8a9ba8', '#238b75'])
        for i, value in enumerate(values):
            ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center')
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f"Same spring test split: {summary['num_samples']} patches")
    fig.tight_layout()
    fig.savefig(output / 'test_metrics.png', dpi=130)
    plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new output folder; existing reports are preserved.')
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')
    checkpoint = load_checkpoint_file(args.best, map_location='cpu')
    config = checkpoint['hyper_parameters']['config']
    config.dataset.root = args.data_root or config.dataset.root
    split, indexed, loader = evaluation.build_dataloader(config, split_override='test', batch_size=1, num_workers=0)
    if not len(indexed):
        raise ValueError('Empty test split')
    paths = get_subset_paths(indexed.dataset, list(range(len(indexed))))
    selected, roi_count = choose_examples(paths, args.seed)
    selection = {'seed': args.seed, 'indices': selected, 'test_roi_count': roi_count,
                 'selection_before_evaluation': True, 'items': []}
    for index in selected:
        selection['items'].append({'index': index,
            'sample_id': evaluation.resolve_sample_name(indexed.dataset, index),
            'paths': {key: str(Path(value).relative_to(config.dataset.root)) for key, value in paths[index].items()}})
    write_json(output / 'selection.json', selection)
    print(f'Selected cases before inference: {selected}; test ROIs={roi_count}', flush=True)
    # Reject exact path overlap with the stored training subset.
    train_paths = json.loads(Path(args.train_manifest).read_text(encoding='utf-8'))['paths']
    normalize = lambda path: str(Path(path).resolve()).casefold()
    overlap = {normalize(p['S1']) for p in paths} & {normalize(p['S1']) for p in train_paths}
    if overlap:
        raise ValueError('Exact training/test file overlap')
    source_files = [Path('train.py'), Path('test.py'), Path('export_test_report.py')]
    for folder in ['models', 'data', 'util', 'config']:
        source_files.extend(Path(folder).rglob('*.py'))
    provenance = {'date': datetime.now().astimezone().isoformat(),
        'evaluation_git_head': subprocess.check_output(['git', '-c', 'safe.directory=C:/CtrS', 'rev-parse', 'HEAD'], text=True).strip(),
        'training_code_commit': None, 'training_code_commit_note': 'Not recorded by historical run',
        'best': {'path': args.best, 'sha256': digest(args.best), 'epoch': checkpoint['epoch']},
        'last': {'path': args.last, 'sha256': digest(args.last)},
        'source_sha256': {str(path): digest(path) for path in source_files},
        'torch': torch.__version__, 'device': str(evaluation.get_device(args.gpu)),
        'input_shape': [15, 256, 256], 'output_channels': 13, 'crop': None,
        'precision': 'float32', 'tta': False, 'metrics_clipped': False,
        'rgb_bands_zero_based': [3, 2, 1], 'png_brightness': 3.0,
        'exact_train_subset_overlap': len(overlap), 'train_subset_size': len(train_paths)}
    del checkpoint
    device = evaluation.get_device(args.gpu)
    model = evaluation.load_model(config, args.best, device)
    rows, cases = [], {}
    with torch.inference_mode(), (output / 'model_stdout.log').open('w') as verbose:
        for step, batch in enumerate(loader):
            index = int(batch['index'][0])
            sample_id = batch['sample_id'][0]
            batch = evaluation.move_batch_to_device(batch, device)
            with contextlib.redirect_stdout(verbose):
                _, merged, target = model.fuse_input(batch)
                prediction = model(merged)[0].float()
            best_metrics = metric_values(prediction, target)
            cloudy_metrics = metric_values(batch['cloudy'], target)
            rows.append({'index': index, 'sample_id': sample_id, **best_metrics,
                         **{'cloudy_' + key: value for key, value in cloudy_metrics.items()}})
            if index in selected:
                cases[index] = {'sample_id': sample_id, 'sar': evaluation.tensor_to_numpy(batch['SAR'])[0],
                    'cloudy': evaluation.tensor_to_numpy(batch['cloudy'])[0],
                    'target': evaluation.tensor_to_numpy(target)[0], 'best': evaluation.tensor_to_numpy(prediction)[0],
                    'best_metrics': best_metrics, 'cloudy_metrics': cloudy_metrics}
            if (step + 1) % 100 == 0:
                print(f'EVALUATION {step + 1}/{len(indexed)}', flush=True)
    # Same five examples only: diagnose the latest failed checkpoint, not full-test performance.
    last_checkpoint = load_checkpoint_file(args.last, map_location='cpu')
    provenance['last']['epoch'] = last_checkpoint['epoch']
    model.load_state_dict(evaluation.extract_state_dict(last_checkpoint), strict=True)
    del last_checkpoint
    selected_metrics = []
    with torch.inference_mode(), (output / 'model_stdout.log').open('a') as verbose:
        for number, index in enumerate(selected, 1):
            case = cases[index]
            merged = torch.from_numpy(np.concatenate([case['sar'], case['cloudy']])[None]).to(device)
            with contextlib.redirect_stdout(verbose):
                last = model(merged)[0].float()
            target = torch.from_numpy(case['target'][None]).to(device)
            case['last'] = evaluation.tensor_to_numpy(last)[0]
            last_metrics = metric_values(last, target)
            selected_metrics.append({'index': index, 'sample_id': case['sample_id'],
                'best': case['best_metrics'], 'last': last_metrics, 'cloudy': case['cloudy_metrics'],
                'best_range': [float(case['best'].min()), float(case['best'].max())],
                'last_range': [float(case['last'].min()), float(case['last'].max())]})
            plot_case(output, number, case)
    summary = {'split': split, 'num_samples': len(rows), 'num_test_rois': roi_count,
        'best_test': {key: float(np.mean([row[key] for row in rows])) for key in METRICS},
        'cloudy_baseline': {key: float(np.mean([row['cloudy_' + key] for row in rows])) for key in METRICS},
        'last_checkpoint_scope': 'Selected five examples only; not a full-test mean'}
    summary['best_minus_cloudy'] = {key: summary['best_test'][key] - summary['cloudy_baseline'][key] for key in METRICS}
    evaluation.write_metrics_csv(output / 'metrics.csv', rows)
    write_json(output / 'summary.json', summary)
    write_json(output / 'selected_metrics.json', selected_metrics)
    write_json(output / 'provenance.json', provenance)
    diagnostics = [json.loads(line) for line in Path(args.diagnostics).read_text(encoding='utf-8').splitlines()]
    compact = [row for row in diagnostics if row['event'] in ('EPOCH_SUMMARY', 'VALIDATION_SUMMARY')]
    write_json(output / 'curves.json', compact)
    plot_curves(output, compact)
    plot_metrics(output, summary)
    artifacts = ['summary.json', 'metrics.csv', 'selection.json', 'selected_metrics.json',
                 'provenance.json', 'curves.json', 'learning.png', 'test_metrics.png']
    artifacts += [f'sample_{number:02d}.png' for number in range(1, len(selected) + 1)]
    write_json(output / 'report_manifest.json', {'complete': True,
        'files_sha256': {name: digest(output / name) for name in artifacts}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--best', required=True)
    parser.add_argument('--last', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    run(parser.parse_args())
