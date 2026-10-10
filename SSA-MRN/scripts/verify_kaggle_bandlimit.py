"""Verify the archived QB K4 B1/B1_AA evidence; optionally load on CPU.

python SSA-MRN/scripts/verify_kaggle_bandlimit.py --torch
No training, network, or source-data access is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/assets/kaggle_bandlimit_k4')
parser.add_argument('--torch', action='store_true', help='Also restore original weights/optimizer/scaler and run rectangular CPU inference')
args = parser.parse_args()
root = args.root
load = lambda p: json.loads(p.read_text())
manifest = load(root / 'report_manifest.json')
for name, record in manifest['files'].items():
    p = root / name
    assert p.is_file(), name
    assert p.stat().st_size == record['bytes'], name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256'], name

runs = sorted((root / 'runs').iterdir())
histories = []
initializations = []
for run in runs:
    history = load(run / 'history.json')
    result = load(run / 'results.json')
    histories.append(history)
    initializations.append(load(run / 'initialization.json')['core_sha256'])
    assert [row['epoch'] for row in history] == list(range(1, 201))
    best = min(history, key=lambda row: row['val_mse'])
    assert (best['epoch'], best['val_mse']) == (result['best_epoch'], result['best_mse'])
    assert load(run / 'training_complete.json')['epoch'] == 200
    for protocol in ('RR', 'FR'):
        rows = load(run / f'{protocol}_per_scene.json')
        assert [row['scene_index'] for row in rows] == list(range(20))
        assert rows == result['metrics'][protocol]['rows']
        for key, mean in result['metrics'][protocol]['mean'].items():
            assert math.isclose(sum(row[key] for row in rows)/20, mean, rel_tol=1e-12)
    for kind, record in result['checkpoints'].items():
        assert hashlib.sha256((run / f'{kind}.pt').read_bytes()).hexdigest() == record['sha256']
assert len(set(initializations)) == 1
assert all(a['sample_order_sha256'] == b['sample_order_sha256'] for a, b in zip(*histories))

if args.torch:
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(root / 'code/CtrS/SSA-MRN/src'))
    spec = importlib.util.spec_from_file_location('archived_bandlimit', root / 'code/B1_AA_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = {}
    for run in runs:
        result = load(run / 'results.json')
        model = module.build_model(result['config']['variant'], 4, 4)
        parameters = sum(p.numel() for p in model.parameters())
        for kind in ('best', 'latest', 'best_first100'):
            ck = torch.load(run / f'{kind}.pt', map_location='cpu', weights_only=True)
            assert ck['config'] == result['config']
            model.load_state_dict(ck['model'], strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.load_state_dict(ck['optimizer'])
            scaler = torch.amp.GradScaler('cpu')
            scaler.load_state_dict(ck['scaler'])
            for key in ('cpu_rng', 'cuda_rng', 'loader_rng', 'python_rng', 'numpy_rng', 'history'):
                assert key in ck
            if kind == 'latest':
                assert ck['epoch'] == 200 and ck.get('mid_epoch') is None
                assert len(ck['history']) == 200 and ck['best_recovery']['epoch'] == result['best_epoch']
            if kind == 'best':
                assert ck['epoch'] == result['best_epoch']
            if kind == 'best_first100':
                assert ck['epoch'] == min(histories[runs.index(run)][:100], key=lambda row: row['val_mse'])['epoch']
        model.eval()
        with torch.inference_mode():
            out = model(torch.zeros(1, 1, 64, 80), torch.zeros(1, 4, 64, 80), torch.zeros(1, 4, 16, 20))
        assert list(out.shape) == [1, 4, 64, 80] and torch.isfinite(out).all()
        summary[run.name] = {'parameters': parameters, 'rectangular_shape': list(out.shape), 'original_checkpoints_loaded': 3}
    print(json.dumps(summary, indent=2))
print(f'PASS: {len(manifest["files"])} archived files; six weights; 400 epochs; 80 scene evaluations; matching initialization and sample order')
