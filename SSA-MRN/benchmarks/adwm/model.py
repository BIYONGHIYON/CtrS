"""ADWM adapter: official code in upstream/ (Jie-1203/ADWM, unmodified); inputs and output in DN / scale.

The paper's proposed model is ADWM on LAGNet ("We incorporated ADWM into the classic LAGNet as our proposed method", Sec. 3.2),
which is upstream/model/bwnet_lagconv_sota.py (BWNet, 48 channels, 6 LAC residual blocks). The official network takes the
low-resolution ms and pan, upsamples ms itself (bicubic, x4) and adds it back as the residual; our lms is not used.
The official test script (main_test.py) clamps the output to [0, 1]; the adapter does the same in eval mode only.

CACW hidden size: the official code fixes it at round(0.8 * channels). The paper uses 0.8n for WV3 and GF2 and 0.6n for QB
(implementation details), so for QB retrain the adapter rebuilds the cross-feature block with 0.6n from the official classes.
The WV3 author weights load with the official 0.8n.

Author weights and LayerNorm: the published weights/BWNET_LAGConv_WV3/model.pth matches the official BWNet key by key and shape
by shape except the LayerNorm inside each CovBlock (self.ln, 14 tensors), which the weight file does not contain. The weights
were evidently trained with a CovBlock without that LayerNorm, so for setting 'author' the adapter replaces each CovBlock.ln
with an identity before a strict load. Leaving a fresh LayerNorm in place (strict=False) would change the computation the
weights were trained with. Whether this reproduces the paper is checked on WV3 RR test (paper: PSNR 39.170, SAM 2.913,
ERGAS 2.145). retrain uses the official code as published, LayerNorm included.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / 'upstream'
_spec = importlib.util.spec_from_file_location('adwm_upstream_lagconv', UPSTREAM / 'model/bwnet_lagconv_sota.py')
_official = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_official)  # official code, unmodified
BWNet, CovBlock = _official.BWNet, _official.CovBlock

CHANNELS, BLOCKS = 48, 6                  # BWNet defaults (paper: 48 hidden channels)
CACW_RATIO = {'QB': 0.6}                  # paper implementation details; other sensors keep the official 0.8


class Adapter(nn.Module):
    def __init__(self, bands, cacw_ratio=0.8, covariance_layernorm=True):
        super().__init__()
        self.net = BWNet(pan_dim=1, ms_dim=bands, channel=CHANNELS, num_lacrbs=BLOCKS)
        if cacw_ratio != 0.8:
            self.net.bw_output.CovBlockList = nn.ModuleList(
                [CovBlock(CHANNELS, 1, round(CHANNELS * cacw_ratio), 0.05) for _ in range(BLOCKS)])
        if not covariance_layernorm:     # layout of the published author weights (see module docstring)
            for block in list(self.net.bw_output.CovBlockList) + [self.net.bw_output.global_covblock]:
                block.ln = nn.Identity()

    def forward(self, pan, lms=None, ms=None):
        out = self.net(ms, pan)          # official input order (ms, pan); lms is not used by ADWM
        return out if self.training else out.clamp(0, 1)


def build(bands, sensor, setting, seed):
    train_sensor = 'WV3' if sensor == 'WV2' else sensor
    if setting == 'author':
        if train_sensor != 'WV3':
            raise ValueError('ADWM author weights exist for WV3 only (WV2 uses them as transfer); QB and GF2 are retrain')
        return Adapter(bands, 0.8, covariance_layernorm=False)
    if setting == 'retrain':
        return Adapter(bands, CACW_RATIO.get(train_sensor, 0.8))
    raise ValueError(f'unknown setting {setting}')


def load(net, path):
    """Author weights are a bare BWNet state_dict; our checkpoints store it under 'model'."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state, strict=True)
