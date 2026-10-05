"""Synthetic full-size CUDA forward/backward audit; no optimizer or training run."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SSA-MRN/src'))
from ssamrn.models.rgb_grouped import build_rgb_hsi_model
from ssamrn.models.area_consistency import project_area_consistency
from ssamrn.losses import reconstruction_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    records = []
    labels = ['00_baseline10', '03_aligned_attention10', '04_gated_detail10', '05_global_encoder10', '06_local_alignment10']
    for label in labels:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        cfg = json.loads((ROOT / 'SSA-MRN/configs/pilot' / (label + '.json')).read_text(encoding='utf-8'))
        model = build_rgb_hsi_model(cfg).cuda().train()
        # Exercise gradients beyond the zero-initialized residual head as well.
        with torch.no_grad():
            model.decoder[-1].weight.normal_(std=.001)
            if getattr(model, 'detail_path', None) is not None:
                model.detail_path.decode.weight.normal_(std=.001)
        gt = torch.rand(4, 204, 256, 256, device='cuda')
        rgb = torch.rand(4, 3, 256, 256, device='cuda')
        lr = F.avg_pool2d(gt, 4)
        mask = torch.ones(4, 256, 256, dtype=torch.bool, device='cuda')
        with torch.autocast('cuda', enabled=True):
            pred = model(rgb, lr)
        pred = project_area_consistency(pred, lr, mask)
        loss, _, _ = reconstruction_loss(pred, gt, mask, .01)
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite CUDA loss/gradient: ' + label)
        item = {'label': label, 'parameters': sum(p.numel() for p in model.parameters()),
                'batch_size': 4, 'patch_size': 256, 'amp': True,
                'loss': loss.item(), 'peak_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mb': torch.cuda.max_memory_reserved() / 2**20,
                'optimizer_steps': 0, 'note': 'Synthetic graph check; Adam state and loader prefetch are not allocated.'}
        records.append(item)
        print(json.dumps(item), flush=True)
        del model, gt, rgb, lr, mask, pred, loss
    torch.cuda.empty_cache()
    args.output.write_text(json.dumps({'training_started': False, 'checks': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
