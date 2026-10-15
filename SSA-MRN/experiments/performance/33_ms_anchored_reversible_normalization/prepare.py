"""Validate retained artifacts and strict-load real weights; never retrains or rescans H5."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-only', action='store_true')
    args = parser.parse_args()
    manifest = read('manifest.json')
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size == expected['bytes'], relative
        assert digest(path) == expected['sha256'], relative
    arms = ('N0', 'N1')
    histories = {arm: read(f'results/{arm}/history.json') for arm in arms}
    assert read('results/N0/initialization.json')['state_sha256'] == read('results/N1/initialization.json')['state_sha256']
    for arm in arms:
        assert [row['epoch'] for row in histories[arm]] == list(range(101))
        assert histories[arm][-1]['updates'] == histories[arm][-1]['attempted'] == 26800
        assert histories[arm][-1]['skipped'] == 0
    assert [row.get('order_sha256') for row in histories['N0']] == [row.get('order_sha256') for row in histories['N1']]
    for arm in arms:
        data = read(f'results/{arm}/metrics.json')
        for protocol in ('RR', 'FR'):
            records = data['protocols'][protocol]
            assert [row['index'] for row in records['rows']] == list(range(20))
            for key, value in records['mean'].items():
                actual = sum(row['metrics'][key] for row in records['rows']) / 20
                assert abs(actual - value) < 1e-12, (arm, protocol, key)
        assert data['best_epoch'] == min(histories[arm], key=lambda row: row['val_mse'])['epoch']
    config = read('provenance/config.json')
    for relative, expected in config['source_hashes'].items():
        assert digest(ROOT / 'provenance/source' / relative) == expected, relative
    print(f'PASS: {len(manifest["files"])} file hashes; real histories, selected epochs, source hashes and scene means')
    if args.hash_only:
        print('Model loading/inference not requested; original H5 not re-evaluated.')
        return
    import sys
    import torch
    sys.path.insert(0, str(ROOT / 'provenance/source'))
    from normalization import Model
    torch.set_num_threads(1)
    torch.manual_seed(123)
    inputs = [torch.rand(shape) * .3 + .1 for shape in ((1, 1, 64, 64), (1, 4, 64, 64), (1, 4, 16, 16))]
    for arm in arms:
        for name in ('initial', 'best', 'latest'):
            path = ROOT / 'checkpoints' / f'{arm}_QB_s42' / f'{name}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['fingerprint'] == config['fingerprint']
            model = Model(arm, config['std_floor'])
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval()
            assert all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in model.state_dict().values())
            with torch.inference_mode():
                prediction = model(*inputs)
            assert prediction.shape == (1, 4, 64, 64) and bool(torch.isfinite(prediction).all())
            if name == 'best':
                assert checkpoint['best_epoch'] == min(histories[arm], key=lambda row: row['val_mse'])['epoch']
            if name == 'latest':
                assert checkpoint['epoch'] == 100 and checkpoint['next_batch'] == 0
                assert checkpoint['updates'] == checkpoint['attempted'] == 26800
                for field in ('optimizer', 'scaler', 'rng', 'generator', 'order', 'best_model', 'history'):
                    assert field in checkpoint, field
                # Compare against the actual selected weight, not the last epoch.
                best = torch.load(path.with_name('best.pt'), map_location='cpu', weights_only=False)
                assert all(torch.equal(value, best['model'][key]) for key, value in checkpoint['best_model'].items())
            print(f'PASS: {arm}/{name}.pt strict load and finite CPU synthetic inference')
    print('No H5 re-evaluation, new training or full CUDA resume test was performed.')


if __name__ == '__main__':
    main()
