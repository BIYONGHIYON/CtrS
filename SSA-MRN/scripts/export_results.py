"""Evaluate saved weights and export graphs + five independent test scenes (4 panels)."""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from report_graphs import compact_history, plot_report
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SSA-MRN/src'))
from ssamrn.data.lib_hsi import LIBHSI


def choose_samples(scene_count, per_scene, seed):
    if scene_count < 5:
        raise ValueError('Five independent test scenes are required')
    scenes = random.Random(seed).sample(range(scene_count), 5)
    return {'seed': seed, 'scene_indices': scenes,
            'sample_indices': [index * per_scene for index in scenes], 'tile': 0,
            'selection': 'Five independent scenes chosen before test metrics; tile 0 fixed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--history', type=Path, help='Defaults to checkpoint folder history.jsonl or curves.json')
    parser.add_argument('--previous-metrics', type=Path)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    # Freeze one weight file so evaluation and all five previews cannot mix epochs.
    snapshot_dir = tempfile.TemporaryDirectory()
    frozen = Path(snapshot_dir.name) / args.checkpoint.name
    shutil.copyfile(args.checkpoint, frozen)
    weight_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
    state = torch.load(frozen, map_location='cpu', weights_only=True)
    config = dict(state['config'])
    if args.device:
        config['device'] = args.device
    config['data_root'] = str(args.data_root.resolve())
    history = args.history or args.checkpoint.parent / 'history.jsonl'
    if not history.exists():
        history = args.checkpoint.parent / 'curves.json'
    curves = compact_history(history)
    if not curves['epochs']:
        parser.error('Measured training history is required')
    manifest = ROOT / config['alignment_manifest'] if config.get('alignment_manifest') else None
    data = LIBHSI(args.data_root, 'test', config['patch_size'], alignment_manifest=manifest,
                  eval_layout=config.get('eval_layout', 'tiles'), degradation=config.get('degradation', 'bicubic'))
    selection = choose_samples(len(data.files), data.per_scene, args.seed)
    selection['scene_ids'] = [data.files[index].stem for index in selection['scene_indices']]
    previous = json.loads(args.previous_metrics.read_text(encoding='utf-8-sig')) if args.previous_metrics else None
    if args.output_dir.exists():
        parser.error('Use a new output directory; existing results are preserved')
    args.output_dir.mkdir(parents=True)
    selection_path = args.output_dir / 'selection.json'
    selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # Temporary config is reconstructed from the weight, not an archived training JSON.
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'evaluation.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        subprocess.run([sys.executable, str(ROOT/'SSA-MRN/scripts/train_lib.py'), '--config', str(path),
                        '--checkpoint', str(frozen), '--data-root', str(args.data_root.resolve()),
                        '--output-dir', str(args.output_dir.resolve()), '--evaluate'], check=True)
    metrics_path = args.output_dir / 'test/metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('partial') or set(metrics['scenes']) != {p.stem for p in data.files}:
        raise ValueError('Incomplete or mismatched full test scene evaluation')
    for i, index in enumerate(selection['sample_indices'], 1):
        subprocess.run([sys.executable, str(ROOT/'SSA-MRN/scripts/preview_lib.py'),
                        '--checkpoint', str(frozen), '--data-root', str(args.data_root.resolve()),
                        '--split', 'test', '--sample-index', str(index),
                        '--output-dir', str((args.output_dir/f'sample_{i:02}').resolve())] +
                       (['--device', args.device] if args.device else []), check=True)
    subprocess.run([sys.executable, str(ROOT/'SSA-MRN/scripts/export_band_viewer.py'),
                    '--checkpoint', str(frozen), '--expected-sha256', weight_hash,
                    '--data-root', str(args.data_root.resolve()),
                    '--output-dir', str((args.output_dir/'band_viewer').resolve()),
                    '--sample-indices', *map(str, selection['sample_indices'])], check=True)
    for i in range(1,6):
        path = args.output_dir/f'sample_{i:02}/metrics.json'
        sample_metrics = json.loads(path.read_text(encoding='utf-8'))
        sample_metrics.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=weight_hash)
        if sample_metrics['epoch'] != state['epoch']:
            raise ValueError('Preview checkpoint epoch mismatch')
        path.write_text(json.dumps(sample_metrics,indent=2),encoding='utf-8')
    (args.output_dir/'curves.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')
    plot_report(curves, metrics, args.output_dir, previous)
    protocol_keys = ('model_type','latent_channels','ssai_dimension','patch_size','train_layout','eval_layout',
                     'degradation','upsampler','alignment_manifest','alignment_sha256','seed','spectral_weight','spectral_eps')
    evidence = {'checkpoint_sha256': weight_hash,
                'checkpoint_source': str(args.checkpoint.resolve()), 'epoch': state['epoch'], 'protocol': {k:config[k] for k in protocol_keys if k in config},
                'test_scenes': len(data.files), 'panel_order': ['LR HSI','RGB guide','Prediction','Ground truth'],
                'complete': True}
    if previous:
        evidence['previous_metrics'] = str(args.previous_metrics)
        evidence['previous_delta'] = {key: metrics['scene_mean'][key]-previous['scene_mean'][key]
                                      for key in ('model_mse','model_psnr_db','model_sam_deg')}
        evidence['comparability'] = 'Check split/IDs, scale, masking, degradation and layouts before claiming improvement'
    for name in ('run_config.json','test/run_config.json'):
        (args.output_dir/name).unlink(missing_ok=True)
    # A complete marker is written only when every required result file exists.
    required = ['learning.png','test_metrics.png','test/metrics.json','curves.json','selection.json']
    required += [f'sample_{i:02}/comparison.png' for i in range(1,6)]
    if not all((args.output_dir/name).is_file() for name in required):
        raise ValueError('Required report assets are missing')
    (args.output_dir/'report_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    snapshot_dir.cleanup()
    print(f'Complete report assets: {args.output_dir}')


if __name__ == '__main__':
    main()
