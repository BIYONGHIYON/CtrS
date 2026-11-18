"""Reproduce the author-normalized inference used for the archived RR/FR arrays.

This standalone runner reconstructs the original notebook inference loop. Data and
checkpoint paths are explicit; it never chooses a checkpoint using test metrics.
Each output is an HWC DN-valued MAT array named ``sr``.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

folder = Path(__file__).resolve().parent
upstream = folder / 'upstream'
sys.path.insert(0, str(upstream / 'models'))
sys.path.insert(0, str(upstream))
from models import ARNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=['QB', 'GF2', 'WV3', 'WV2'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    bands = 8 if args.sensor in ('WV3', 'WV2') else 4
    hw_range = [1, 9] if args.sensor == 'QB' else [1, 18]
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model']
    net = ARNet(1, bands)
    net.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    net.to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, 'r') as data, torch.inference_mode():
        count = len(data['pan']) if args.limit is None else min(args.limit, len(data['pan']))
        for i in range(count):
            pan = torch.as_tensor(np.asarray(data['pan'][i:i+1], dtype='float32') / 2047, device=args.device)
            lms = torch.as_tensor(np.asarray(data['lms'][i:i+1], dtype='float32') / 2047, device=args.device)
            out = net(pan, lms, 1000, hw_range)[0].cpu().numpy() * 2047
            if out.shape != tuple(lms.shape[1:]) or not np.isfinite(out).all():
                raise ValueError(f'invalid prediction for scene {i}')
            scipy.io.savemat(args.output / f'output_mulExm_{i}.mat', {'sr': np.moveaxis(out, 0, -1)})
            print(f'{args.sensor} scene {i}: saved', flush=True)


if __name__ == '__main__':
    main()
