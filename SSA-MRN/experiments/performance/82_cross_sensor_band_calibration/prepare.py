"""82: per-band sensor calibration for a model applied to a sensor it was not trained on, and the reference statistics of the training sensors.

Calibration (fixed before the evaluation; input statistics only, no ground truth):
  r_train[b] = sum over the training set of LMS_b / sum of PAN (ratio of means, from the training H5 of the training sensor)
  r_scene[b] = mean(LMS_b) / mean(PAN) of the input scene of the target sensor
  t[b] = r_train[b] / r_scene[b]; the network sees (PAN, t * LMS, t * MS); its output is divided by t; then the observation-consistency
  projection with the target sensor's Wald operator (K10, prepare of experiment 83).
Variants (reported, not chosen): 'const' uses one r_target per band (mean over the 20 target test scenes) instead of r_scene; 'level' scales
PAN and every band so that the channel means equal the training-set means. Never applied to the sensor a model was trained on.

`python prepare.py --stats <sensor> <train.h5> <out.json>` writes the training-set statistics (chunked read, CPU only)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # SSA-MRN root for common/
_spec = importlib.util.spec_from_file_location('exp83_prepare', HERE.parent / '83_factorized_c2ms_finetuning/prepare.py')
P83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P83)
SENSORS, data_files, Wald, Projected = P83.SENSORS, P83.data_files, P83.Wald, P83.Projected


def training_stats(path, chunk=1024):
    """Ratio of channel means over the whole training set and the per-patch median ratio (sensitivity only)."""
    import h5py
    with h5py.File(path, 'r') as f:
        n = len(f['pan']); sp = 0.; sb = None; per = []
        for s in range(0, n, chunk):
            pan = np.asarray(f['pan'][s:s + chunk], dtype=np.float64); lms = np.asarray(f['lms'][s:s + chunk], dtype=np.float64)
            sp += pan.sum(); b = lms.sum((0, 2, 3)); sb = b if sb is None else sb + b
            per.append(lms.mean((2, 3)) / np.maximum(pan.mean((1, 2, 3)), 1e-6)[:, None])
    per = np.concatenate(per); px = n * pan.shape[-1] * pan.shape[-2]
    return dict(patches=n, pan_mean_dn=sp / px, band_mean_dn=(sb / px).tolist(), ratio_of_means=(sb / sp).tolist(), patch_median_ratio=np.median(per, 0).tolist())


class Calibrated(torch.nn.Module):
    """Normalized (pan, lms, ms) in (target DN / target max), normalized output out. mode: 'scene' | 'const' | 'level'."""

    def __init__(self, net, ref, mode='scene', target_ratio=None, scales=None):
        super().__init__(); self.net, self.mode = net, mode
        self.r_train = torch.tensor(ref['ratio_of_means'], dtype=torch.float64)
        self.target_ratio = None if target_ratio is None else torch.tensor(target_ratio, dtype=torch.float64)
        self.scales = scales  # (train max, target max, training PAN mean DN, training band means DN) for 'level'

    def forward(self, pan, lms, ms):
        mp = pan.double().mean((1, 2, 3)); mb = lms.double().mean((2, 3))
        if self.mode == 'scene': t = self.r_train[None] * mp[:, None] / mb; kp = torch.ones_like(mp)
        elif self.mode == 'const': t = (self.r_train / self.target_ratio)[None].expand_as(mb); kp = torch.ones_like(mp)
        elif self.mode == 'level':
            smax_train, smax_target, pan_ref, band_ref = self.scales
            kp = (pan_ref / smax_train) / mp; t = torch.tensor(band_ref, dtype=torch.float64)[None] / smax_train / mb
        else: raise ValueError(self.mode)
        t4 = t[..., None, None].to(lms.dtype); y = self.net(pan * kp[:, None, None, None].to(pan.dtype), lms * t4, ms * t4)
        return y / t4


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--stats':
        res = dict(sensor=sys.argv[2], source=sys.argv[3], **training_stats(sys.argv[3])); Path(sys.argv[4]).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    else: raise SystemExit(__doc__)
