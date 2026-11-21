"""DLPan FusionNet adapter: the official network predicts a residual."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / 'upstream' / '01-DL-toolbox(Pytorch)'
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location('dlpan_fusionnet', UPSTREAM / 'UDL/pansharpening/models/FusionNet/model_fusionnet.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Adapter(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.net = module.FusionNet(bands, None)

    def forward(self, pan, lms, ms):
        return lms + self.net(lms, pan)


def build(bands, sensor, setting, seed):
    return Adapter(bands)


def load(net, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
