"""ADWM adapter: official code in upstream/ (Jie-1203/ADWM, unmodified); inputs and output are DN / scale.

Model: the paper's proposed method is ADWM on LAGNet ("We incorporated ADWM into the classic LAGNet as our proposed method",
Sec. 3.2), which is upstream/model/bwnet_lagconv_sota.py (BWNet: 48 channels, 6 LAConv residual blocks, IFW/CFW weighting).
The official network takes the low-resolution ms and pan, upsamples ms itself (bicubic x4) and adds it back as the residual,
so our lms is not used. The official test script clamps the output to [0, 1]; the adapter does the same in eval mode only.

CACW intermediate size d (paper, App. D): 0.8n for WV3 and GF2 (the official code's fixed value), 0.6n for QB. For QB the
adapter replaces the six intra-feature CovBlocks with official CovBlocks of hidden size round(0.6 * 48) = 29.

Published WV3 weights (weights/BWNET_LAGConv_WV3/model.pth) match the official BWNet key by key and shape by shape except the
LayerNorm inside each CovBlock (7 blocks, 14 tensors), which the file does not contain. Two readings fit the file:
    'none'            the weights were trained without that LayerNorm (identity)
    'parameter_free'  they were trained with a LayerNorm without parameters (elementwise_affine=False), which keeps no keys
Both load strictly. The reading used for setting 'author' is chosen on the WV3 validation set (lowest L1), never on test:
    python SSA-MRN/benchmarks/adwm/model.py --select-author --val <valid_wv3.h5> --device cuda
writes results/author_selection.json, which build() reads. retrain uses the official code as published (LayerNorm trained).

Memory: one LAConv layer unfolds its input into B x 9C x H x W tensors, so a training batch of 64 needs more than a 15 GB T4.
set_checkpointing(True) recomputes each LAConv layer in the backward pass (torch.utils.checkpoint, non-reentrant) instead of
storing its intermediates. LAConv layers contain no BatchNorm or dropout, so the loss, the gradients and the BatchNorm
statistics are those of the full batch of 64; only time is spent.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
_spec = importlib.util.spec_from_file_location('adwm_bwnet_lagconv_sota', UPSTREAM / 'model/bwnet_lagconv_sota.py')
official = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(official)  # official code, unmodified

CHANNELS, BLOCKS, DROPOUT = 48, 6, 0.05          # BWNet defaults; CovBlock dropout as in BandSelectBlock
CACW_RATIO = {'QB': 0.6, 'GF2': 0.8, 'WV3': 0.8}  # paper App. D (GF2: "identical to the WV3 configuration")
TRAINING_SENSOR = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}
BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8, 'WV2': 8}
AUTHOR_WEIGHTS = UPSTREAM / 'weights/BWNET_LAGConv_WV3/model.pth'
AUTHOR_SELECTION = HERE / 'results/author_selection.json'
READINGS = ('none', 'parameter_free')
KST = timezone(timedelta(hours=9))


def _checkpointed(module, x):
    forward = type(module).forward
    return checkpoint(forward, module, x, use_reentrant=False) if torch.is_grad_enabled() else forward(module, x)


class ADWM(nn.Module):
    def __init__(self, bands, cacw_ratio=0.8, layernorm='trained'):
        super().__init__()
        self.net = official.BWNet(pan_dim=1, ms_dim=bands, channel=CHANNELS, num_lacrbs=BLOCKS)
        bw = self.net.bw_output
        if cacw_ratio != 0.8:
            bw.CovBlockList = nn.ModuleList(
                [official.CovBlock(CHANNELS, 1, round(CHANNELS * cacw_ratio), DROPOUT) for _ in range(BLOCKS)])
        if layernorm != 'trained':
            if layernorm not in READINGS: raise ValueError(f'layernorm must be trained, none or parameter_free, not {layernorm}')
            for block in list(bw.CovBlockList) + [bw.global_covblock]:
                dim = block.ln.normalized_shape
                block.ln = nn.Identity() if layernorm == 'none' else nn.LayerNorm(dim, elementwise_affine=False)
        self.layernorm, self.cacw_ratio = layernorm, cacw_ratio

    def set_checkpointing(self, enabled):
        for m in self.net.modules():
            if isinstance(m, official.LAConv2D):
                if enabled: m.forward = partial(_checkpointed, m)
                else: m.__dict__.pop('forward', None)
        return self

    def forward(self, pan, lms=None, ms=None):
        out = self.net(ms, pan)              # official input order (ms, pan)
        return out if self.training else out.clamp(0, 1)


def author_reading():
    if not AUTHOR_SELECTION.is_file():
        raise RuntimeError(f'{AUTHOR_SELECTION} is missing: choose the LayerNorm reading on WV3 validation first '
                           '(python SSA-MRN/benchmarks/adwm/model.py --select-author --val <valid_wv3.h5>)')
    return json.loads(AUTHOR_SELECTION.read_text(encoding='utf-8'))['chosen']


def build(bands, sensor, setting, seed):
    train = TRAINING_SENSOR[sensor]
    if setting == 'author':
        if train != 'WV3': raise ValueError('ADWM author weights exist for WV3 only (WV2 uses them as transfer); QB and GF2 are retrain')
        return ADWM(bands, 0.8, author_reading())
    if setting == 'retrain':
        return ADWM(bands, CACW_RATIO[train])
    raise ValueError(f'unknown setting {setting}')


def load(net, path):
    """Author weights are a bare BWNet state_dict, our checkpoints keep it under 'model'. Always a strict load."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    net.net.load_state_dict(state['model'] if 'model' in state else state, strict=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def validation_l1(net, path, sensor, device, batch=32, limit=None):
    """Mean absolute error over every validation pixel, unclamped output in eval mode (as the official validation loss)."""
    import h5py
    import numpy as np
    scale = 1023. if sensor == 'GF2' else 2047.
    net.to(device).eval(); total, count = 0.0, 0
    with h5py.File(path, 'r') as h, torch.no_grad():
        n = h['gt'].shape[0] if limit is None else min(limit, h['gt'].shape[0])
        for i in range(0, n, batch):
            x = {k: torch.from_numpy(np.asarray(h[k][i:min(n, i + batch)], dtype=np.float32) / scale).to(device) for k in ('pan', 'ms', 'gt')}
            out = net.net(x['ms'], x['pan'])
            total += (out - x['gt']).abs().sum().item(); count += x['gt'].numel()
    return total / count, n


def select_author(val, device, limit=None):
    """Evaluate both readings of the WV3 author weights on WV3 validation and record the one with the lower L1."""
    rows = {}
    for reading in READINGS:
        net = ADWM(8, 0.8, reading); load(net, AUTHOR_WEIGHTS)
        rows[reading], n = validation_l1(net, val, 'WV3', device, limit=limit)
        print(f'{reading:15s} validation L1 {rows[reading]:.7f} ({n} samples)', flush=True)
    out = dict(weights=AUTHOR_WEIGHTS.relative_to(HERE).as_posix(), weights_sha256=sha256(AUTHOR_WEIGHTS),
               data=dict(file=Path(val).name, sha256=sha256(val), samples=n, limit=limit),
               criterion='WV3 validation L1 over all pixels, unclamped output in eval mode (official validation loss); test is not used',
               candidates={r: dict(val_l1=v) for r, v in rows.items()}, chosen=min(rows, key=rows.get),
               device=str(device), created=datetime.now(KST).isoformat(timespec='seconds'))
    if limit is None:
        AUTHOR_SELECTION.parent.mkdir(parents=True, exist_ok=True)
        AUTHOR_SELECTION.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print('wrote', AUTHOR_SELECTION, '-> chosen', out['chosen'], flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--select-author', action='store_true', help='choose the LayerNorm reading of the WV3 author weights on validation')
    p.add_argument('--val', type=Path, help='WV3 validation H5 (valid_wv3.h5)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, help='smoke test on the first N samples (nothing is written)')
    a = p.parse_args()
    if not a.select_author or not a.val: p.error('use --select-author --val <valid_wv3.h5>')
    select_author(a.val, a.device, a.limit)
