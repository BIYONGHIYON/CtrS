"""Validation-only RGB reliance diagnosis, with a common conservative mask."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SSA-MRN/src'))
from train_lib import evaluate, worker_init
from ssamrn.data.lib_hsi import LIBHSI
from ssamrn.models.rgb_grouped import build_rgb_hsi_model
from ssamrn.models.interp23 import interp23tap


class CommonMask(Dataset):
    def __init__(self, dataset, margin=4):
        self.dataset, self.margin = dataset, margin

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        mask = item.get('valid_mask', torch.ones(item['gt'].shape[-2:], dtype=torch.bool))
        # Includes invalid registration boundaries and all diagnostic shifts.
        size = 2 * self.margin + 1
        invalid = F.pad((~mask).float()[None, None], (self.margin,) * 4, value=1)
        item['valid_mask'] = F.max_pool2d(invalid, size, stride=1)[0, 0].eq(0)
        return item


def perturb_rgb(rgb, variant):
    if variant == 'original':
        return rgb
    if variant == 'blur_x4':
        return interp23tap(F.avg_pool2d(rgb, 4), 4)
    if variant == 'mean_rgb':
        return rgb.mean((-2, -1), keepdim=True).expand_as(rgb)
    if variant == 'zero_rgb':
        return torch.zeros_like(rgb)
    if variant in ('shift_left1', 'shift_right1'):
        padded = F.pad(rgb, (1, 1, 0, 0), mode='replicate')
        start = 2 if variant == 'shift_left1' else 0
        return padded[..., start:start + rgb.shape[-1]]
    raise ValueError('Unknown RGB diagnostic: ' + variant)


class RGBVariantBatches:
    def __init__(self, loader, variant):
        self.loader, self.variant = loader, variant

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield dict(batch, rgb=perturb_rgb(batch['rgb'], self.variant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    checkpoint = Path(config['diagnostic_checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved = state['config']
    alignment = ROOT / config['alignment_manifest']
    if hashlib.sha256(alignment.read_bytes()).hexdigest() != saved.get('alignment_sha256'):
        parser.error('Diagnostic alignment must match checkpoint')
    for key in ('degradation', 'upsampler', 'patch_size', 'eval_layout'):
        if config[key] != saved[key]:
            parser.error('Diagnostic protocol differs: ' + key)
    torch.set_num_threads(config.get('cpu_threads', 2))
    device = torch.device(args.device or config['device'])
    model = build_rgb_hsi_model(saved).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    dataset = LIBHSI(root=ROOT / config['data_root'], split='validation', patch_size=256,
                     patches_per_scene=4, train_layout='quadrants', eval_layout='tiles',
                     degradation='area', compute_lr=True, read_mode='stripes', alignment_manifest=alignment)
    loader = DataLoader(CommonMask(dataset), batch_size=config['val_batch_size'],
                        num_workers=config['workers'], pin_memory=device.type == 'cuda',
                        worker_init_fn=worker_init)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'split': 'validation', 'scenes_count': len(dataset.files), 'tiles_count': len(dataset),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'common_mask_margin': 4, 'variants': {},
              'note': 'Zero/mean guides are out-of-distribution controls; this measures sensitivity, not a retrained HSI-only baseline.'}
    for variant in ('original', 'blur_x4', 'shift_left1', 'shift_right1', 'mean_rgb', 'zero_rgb'):
        output = args.output_dir / variant
        output.mkdir(exist_ok=True)
        metrics = evaluate(model, RGBVariantBatches(loader, variant), device, config.get('amp', False) and device.type == 'cuda',
                           output, baseline_mode='23tap', degradation_mode='area',
                           area_consistency=True, spatial_metrics=True)
        metrics['split'] = 'validation'
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        result['variants'][variant] = metrics['scene_mean']
        (args.output_dir / 'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'variant': variant, **metrics['scene_mean']}), flush=True)
    base = result['variants']['original']
    result['delta_from_original'] = {key: {metric: value[metric] - base[metric] for metric in base}
                                     for key, value in result['variants'].items()}
    result['complete'] = True
    (args.output_dir / 'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
