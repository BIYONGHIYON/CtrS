"""ADWM adapter: the official LAGNet + ADWM network (upstream/model/bwnet_lagconv_sota.py, unmodified).

Inputs and output are DN / scale (QB, WV3, WV2 2047; GF2 1023), the same normalization as the official data.py.
The official network takes (ms, pan) and upsamples ms by bicubic interpolation inside; lms is not used.
At inference the output is clipped to 0-1 as in the official main_test.py. Training uses the unclipped output (main_train.py).

Settings (config.json):
- retrain: official class as published, with LayerNorm. QB uses the IFW hidden size 0.6n of the paper (supplement D);
  the official code fixes 0.8n, so the six IFW CovBlocks are rebuilt here with round(48 * 0.6) = 29. GF2 and WV3 keep 0.8n.
- author: the published WV3 weights have no LayerNorm tensors (14 missing keys). The handling ('identity' = no LayerNorm,
  'default' = LayerNorm left at weight 1, bias 0) is chosen on WV3 validation (train.py author-select) and fixed in config.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
spec = importlib.util.spec_from_file_location('adwm_bwnet_lagconv_sota', UPSTREAM / 'model' / 'bwnet_lagconv_sota.py')
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)

CHANNELS = 48
IFW_RATIO = {'QB': 0.6, 'GF2': 0.8, 'WV3': 0.8}   # paper supplement D (official code: 0.8 for all)
TRAINING_SENSOR = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}
AUTHOR_WEIGHTS = UPSTREAM / 'weights' / 'BWNET_LAGConv_WV3' / 'model.pth'
LAYERNORM_MODES = ('identity', 'default')


class Adapter(nn.Module):
    def __init__(self, bands, ifw_ratio=0.8, layernorm='keep'):
        super().__init__()
        self.net = upstream.BWNet(ms_dim=bands)
        if ifw_ratio != 0.8:
            blocks = self.net.bw_output.CovBlockList
            for i in range(len(blocks)):
                blocks[i] = upstream.CovBlock(CHANNELS, 1, round(CHANNELS * ifw_ratio), 0.05)
        if layernorm == 'identity':
            for block in covblocks(self.net): block.ln = nn.Identity()
        self.layernorm = layernorm

    def forward(self, pan, lms, ms):
        return torch.clamp(self.net(ms, pan), 0, 1)


def covblocks(net):
    return list(net.bw_output.CovBlockList) + [net.bw_output.global_covblock]


def author_layernorm():
    mode = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))['settings']['author']['layernorm']
    if mode not in LAYERNORM_MODES:
        raise SystemExit(f'config.json settings.author.layernorm is {mode!r}: run train.py author-select on WV3 validation first')
    return mode


def build(bands, sensor, setting, seed, layernorm=None):
    train = TRAINING_SENSOR[sensor]
    if setting == 'author':
        if train != 'WV3': raise SystemExit('author weights exist for WV3 only (WV2 uses them as transfer)')
        return Adapter(bands, 0.8, layernorm or author_layernorm())
    if setting == 'retrain':
        return Adapter(bands, IFW_RATIO[train], 'keep')
    raise SystemExit(f'unknown setting {setting!r}')


def load(net, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state: state = state['model']
    if net.layernorm == 'default':   # author weights: only the 14 LayerNorm tensors may be missing; they keep weight 1, bias 0
        result = net.net.load_state_dict(state, strict=False)
        expected = {k for k in net.net.state_dict() if '.ln.' in k}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise RuntimeError(f'unexpected key mismatch: missing {result.missing_keys}, unexpected {result.unexpected_keys}')
    else:
        net.net.load_state_dict(state, strict=True)
