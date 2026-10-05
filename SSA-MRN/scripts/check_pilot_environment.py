"""Read-only pilot data/checkpoint/environment audit; no training or CUDA allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SSA-MRN/src'))
from ssamrn.data.lib_hsi import LIBHSI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder = ROOT / 'SSA-MRN/configs/pilot'
    config = json.loads((folder / '01_rgb_diagnostic.json').read_text(encoding='utf-8-sig'))
    alignment = ROOT / config['alignment_manifest']
    checkpoint = Path(config['diagnostic_checkpoint'])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = '5e126b17f0d95f445e8226fce401d0891593b245b737e73dc2810a3e4510d704'
    if sha != expected:
        raise RuntimeError('RGB07 best checkpoint checksum differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if hashlib.sha256(alignment.read_bytes()).hexdigest() != state['config']['alignment_sha256']:
        raise RuntimeError('Registration checksum differs')
    scenes = {}
    ids = {}
    for split in ('train', 'validation', 'test'):
        dataset = LIBHSI(root=ROOT / config['data_root'], split=split, patch_size=256, patches_per_scene=4,
                         train_layout='quadrants', eval_layout='tiles', degradation='area',
                         read_mode='stripes', alignment_manifest=alignment)
        scenes[split] = len(dataset.files)
        ids[split] = {file.stem for file in dataset.files}
    if any(ids[a] & ids[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('Scene IDs overlap across splits')
    result = {'python': sys.executable, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'checkpoint_epoch': state['epoch'], 'checkpoint_sha256': sha,
              'alignment_sha256': hashlib.sha256(alignment.read_bytes()).hexdigest(), 'scenes': scenes,
              'jobs_count': len(json.loads((folder / 'suite.json').read_text(encoding='utf-8-sig'))['jobs']),
              'training_started': False, 'note': 'Test file inventory checked; no test prediction or metric computed.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
