"""ADWM adapter: official BWNet with LAGConv (ADWM aa185a6, model/bwnet_lagconv_sota.py).

The official network takes (ms, pan), upsamples ms bicubically inside and returns the fused image; main_test.py clamps the
output to [0, 1] before saving, so the adapter does the same. QB uses the CACW hidden ratio 0.6 of the paper (appendix D) exactly
as train.py builds it; GF2 and WV3 keep the official 0.8. The published WV3 weights are evaluated with the CovBlock LayerNorm
replaced by identity (config.json author.<sensor>.layernorm), the variant selected on WV3 validation for the archived author
predictions; the retrained weights keep the LayerNorm they were trained with. einops (used only for one rearrange) is replaced
by the equivalent flatten/transpose when it is not installed (repeat is imported but never called)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if 'einops' not in sys.modules:
    try:
        import einops  # noqa: F401
    except ImportError:
        shim = types.ModuleType('einops')

        def rearrange(x, pattern, **_):
            if pattern.replace(' ', '') != 'BCHW->B(HW)C': raise NotImplementedError(pattern)
            return x.flatten(2).transpose(1, 2)
        def repeat(*_, **__): raise NotImplementedError('einops.repeat is imported but not used by bwnet_lagconv_sota')
        shim.rearrange = rearrange; shim.repeat = repeat; sys.modules['einops'] = shim
_spec = importlib.util.spec_from_file_location('adwm_bwnet_lagconv', HERE / 'upstream/model/bwnet_lagconv_sota.py')
official = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(official)
TRAINING = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}


class Adapter(nn.Module):
    def __init__(self, bands, training_sensor, layernorm='layernorm'):
        super().__init__()
        self.net = official.BWNet(ms_dim=bands)
        if training_sensor == 'QB':  # train.py: CACW hidden ratio 0.6 for QB
            blocks = self.net.bw_output.CovBlockList
            for i in range(len(blocks)): blocks[i] = official.CovBlock(48, 1, round(48 * 0.6), 0.05)
        self.layernorm = layernorm
        if layernorm == 'identity':
            for m in self.net.modules():
                if isinstance(m, official.CovBlock): m.ln = nn.Identity()
        elif layernorm != 'layernorm': raise ValueError(layernorm)

    def forward(self, pan, lms, ms):
        return torch.clamp(self.net(ms, pan), 0, 1)


def build(bands, sensor, setting, seed):
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    ln = cfg['author'].get(sensor.lower(), {}).get('layernorm', 'layernorm') if setting == 'author' else 'layernorm'
    return Adapter(bands, TRAINING[sensor], ln)


def load(net, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if net.layernorm == 'identity': sd = {k: v for k, v in sd.items() if '.ln.' not in k}
    net.net.load_state_dict(sd, strict=True)
