"""Calibrate and sequentially run validation-only RGB07 gradient-loss trials."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = 'C:/CtrS/.venv/Scripts/python.exe'
CTL = ROOT / 'scripts/remote-training-warm07.py'


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def control(*args):
    result = subprocess.run([PYTHON, str(CTL), *args], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def prepare(source, output):
    import importlib.util
    import torch
    from torch.utils.data import default_collate
    spec = importlib.util.spec_from_file_location('train_lib', ROOT/'SSA-MRN/scripts/train_lib.py')
    lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
    torch.set_num_threads(2)
    state = torch.load(source, map_location='cpu', weights_only=True)
    config = dict(state['config'])
    assert config['model_type'] == 'rgb_triple_joint17_23tap'
    assert not config.get('pilot_options') and config.get('train_area_consistency')
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/'init-source.pt'
    import shutil
    shutil.copy2(source, frozen)
    model = lib.build_rgb_hsi_model(config).cuda().eval()
    model.load_state_dict(state['model'])
    data = lib.LIBHSI(config['data_root'], 'train', patch_size=256, patches_per_scene=4,
                     compute_lr=False, crop_seed=42, read_mode='stripes',
                     alignment_manifest=ROOT/config['alignment_manifest'], train_layout='quadrants',
                     eval_layout='tiles', degradation='area')
    # Eight independent training scenes, four quadrants each; no validation/test calibration.
    scenes = [0, 56, 112, 168, 224, 280, 336, 392]
    mse = gradient = 0.
    with torch.no_grad():
        for scene in scenes:
            batch = default_collate([data[(scene*4+i, 0)] for i in range(4)])
            rgb, lr, gt = lib.move_batch(batch, torch.device('cuda'))
            mask = batch['valid_mask'].cuda()
            with torch.autocast('cuda', enabled=True):
                pred = model(rgb, lr)
            pred = lib.project_area_consistency(pred, lr, mask)
            _, ml, _ = lib.reconstruction_loss(pred, gt, mask, config['spectral_weight'], config['spectral_eps'])
            gl = lib.masked_gradient_l1(pred, gt, mask)
            mse += ml.item(); gradient += gl.item()
    assert gradient > 0 and mse > 0
    paths = []
    for name, ratio in [('control', 0.), ('gradient025', .025), ('gradient05', .05), ('gradient10', .10), ('gradient20', .20)]:
        cfg = dict(config)
        cfg.update(epochs=10, learning_rate=1e-5, init_checkpoint=str(frozen),
                   gradient_weight=ratio*mse/gradient, pilot_metrics=True,
                   output_dir='SSA-MRN/experiments/checkpoints/gradient_suite_placeholder')
        cfg.pop('initialization', None)
        cfg['gradient_calibration'] = {'initial_loss_ratio': ratio, 'mse': mse/8,
                                       'gradient_l1': gradient/8, 'train_scene_indices': scenes}
        path = output/(name+'.json'); write(path, cfg); paths.append(str(path))
    write(output/'suite.json', {'configs': paths, 'source': str(source),
          'source_epoch': state['epoch'], 'source_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
          'created_at': dt.datetime.now().isoformat(), 'selection_split': 'validation', 'runs': []})
    print(str(output/'suite.json'))


def worker(suite_path):
    suite = json.loads(suite_path.read_text())
    status_path = suite_path.parent/'status.json'
    try:
        assert not suite['runs'], 'Refuse to duplicate an already started suite'
        for config in suite['configs']:
            state = control('status')
            if not state.get('controller_online') or state.get('status') in ('running', 'starting'):
                raise RuntimeError('Controller offline or another run is active')
            started = control('start', '--config', config)
            run = started['run']; suite['runs'].append(run); write(suite_path, suite)
            write(status_path, {'status': 'running', 'current': config, 'run': run})
            while True:
                time.sleep(5)
                state = control('status')
                if state.get('id') != run['id'] or not state.get('controller_online'):
                    raise RuntimeError('Run ownership changed or controller went offline')
                if state['status'] not in ('running', 'starting'):
                    if state['status'] != 'finished' or state.get('exit_code') != 0:
                        raise RuntimeError('Training failed: '+json.dumps(state))
                    break
            metrics = Path(state['output'])/'best_validation_metrics.json'
            best = json.loads(metrics.read_text())['scene_mean']
            suite['runs'][-1].update(state); suite['runs'][-1]['best_validation'] = best
            write(suite_path, suite)
        write(status_path, {'status': 'finished', 'runs': suite['runs']})
    except Exception as exc:
        write(status_path, {'status': 'failed', 'error': str(exc), 'runs': suite['runs']})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--suite', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.source, args.output)
    else: worker(args.suite)
