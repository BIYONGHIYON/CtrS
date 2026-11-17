"""ARConv author-weight adapter, inputs and outputs in DN / team sensor scale."""
import sys
from pathlib import Path
import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / 'upstream'
sys.path.insert(0, str(UPSTREAM / 'models'))
sys.path.insert(0, str(UPSTREAM))
from models import ARNet


class Adapter(nn.Module):
    def __init__(self, bands, sensor):
        super().__init__()
        self.net = ARNet(1, bands)
        self.team_scale = 1023.0 if sensor == 'GF2' else 2047.0
        self.hw_range = [1, 9] if sensor == 'QB' else [1, 18]

    def forward(self, pan, lms, ms):
        ratio = self.team_scale / 2047.0
        return self.net(pan * ratio, lms * ratio, 1000, self.hw_range) / ratio


def build(bands, sensor, setting, seed):
    if setting != 'author':
        raise ValueError('only author weights have been evaluated')
    return Adapter(bands, sensor)


def load(net, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    net.net.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
