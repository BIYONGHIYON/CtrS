"""Compare author input scaling and the benchmark adapter on real GF2 scenes.

Run with the pinned upstream checked out at ../upstream:
  python verify_gf2.py --data test_gf2_multiExm1.h5 --checkpoint checkpoint.pth
"""
import argparse
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--scenes', type=int, default=3)
args = parser.parse_args()
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('arconv_adapter', Path(__file__).resolve().parents[1] / 'model.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
net = adapter.build(4, 'GF2', 'author', None).eval()
adapter.load(net, args.checkpoint)
results = []
with h5py.File(args.data, 'r') as data, torch.inference_mode():
    for i in range(args.scenes):
        pan = np.asarray(data['pan'][i:i+1], dtype=np.float32)
        lms = np.asarray(data['lms'][i:i+1], dtype=np.float32)
        gt = np.asarray(data['gt'][i], dtype=np.float64)
        direct = net.net(torch.from_numpy(pan/2047), torch.from_numpy(lms/2047), 1000, [1,18])[0].numpy()*2047
        team_pan = torch.from_numpy((pan.astype(np.float64)/1023).astype(np.float32))
        team_lms = torch.from_numpy((lms.astype(np.float64)/1023).astype(np.float32))
        via = net(team_pan, team_lms, None)[0].numpy()*1023
        diff = np.abs(direct-via)
        psnr = lambda x: float(10*np.log10(2047**2 / np.mean((x.astype(np.float64)-gt)**2)))
        results.append(dict(scene=i, mae_DN=float(diff.mean()), max_DN=float(diff.max()), psnr_direct=psnr(direct), psnr_adapter=psnr(via)))
print(json.dumps(results, indent=2))
