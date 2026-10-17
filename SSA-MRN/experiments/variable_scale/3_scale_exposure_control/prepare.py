"""Verify retained artifacts or reconstruct a fresh original-layout resume source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf8'))
    for name, info in manifest['files'].items():
        path = ROOT / name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != info['bytes'] or digest != info['sha256']:
            raise ValueError('Artifact mismatch: ' + name)
    return len(manifest['files'])


def restore(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError('Use a fresh runtime destination outside this experiment folder')
    verify()
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {'config.json':'settings.json', 'provenance/suite_identity.json':'suite_identity.json',
               'provenance/data_manifest.json':'data_manifest.json',
               'provenance/selection.json':'selection.json', 'provenance/gpu_queues.json':'gpu_queues.json'}
    for source, target in mapping.items():
        shutil.copyfile(ROOT / source, destination / target)
    shutil.copytree(ROOT / 'provenance/code', destination / 'code')
    for folder in sorted((ROOT / 'checkpoints').iterdir()):
        job = folder.name.removesuffix('_QB_s42')
        run = destination / 'runs' / job
        run.mkdir(parents=True)
        for name in ['best.pt','latest.pt']:
            shutil.copyfile(folder / name, run / name)
        for name in ['identity.json','initialization.json']:
            shutil.copyfile(ROOT / 'provenance' / job / name, run / name)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-output', type=Path,
                        help='Fresh original-layout suite directory, for example /kaggle/working/restore_source/seed_42')
    args = parser.parse_args()
    if args.restore_output:
        print('Restored original-layout resume source:', restore(args.restore_output))
        print('Requires original H5 bytes and compatible GPU/software. No training or H5 evaluation was run.')
    else:
        print('PASS:', verify(), 'retained file hashes; no H5 metrics recomputed')
