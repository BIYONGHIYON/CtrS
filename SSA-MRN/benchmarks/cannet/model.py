"""CANNet adapter: official code in upstream/ (Duanyll/CANConv, unmodified); inputs and output in DN / scale.

The official network takes (pan, lms) and returns lms + residual; ms is not used. In eval mode the k-means layers ignore the
training cache and cluster every image from scratch, as in the official test script. The native operators (pwacnative, libKMCUDA)
must be built in upstream/ first (upstream/build.sh, CUDA only).
"""
import sys
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / 'upstream'
sys.path.insert(0, str(UPSTREAM))
from canconv.models.cannet.model import CANNet  # noqa: E402  (official code, unmodified)

CHANNELS, CLUSTERS, THRESHOLD = 32, 32, 0.005  # canconv/models/cannet/default.json


class Adapter(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.net = CANNet(bands, CHANNELS, CLUSTERS, THRESHOLD)

    def forward(self, pan, lms, ms=None, index=None):
        return self.net(pan, lms, index)  # official input order; index selects the k-means cache during training only


def build(bands, sensor, setting, seed):
    return Adapter(bands)


def load(net, path):
    """Author weights are a bare CANNet state_dict; our checkpoints store it under 'model'."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state, strict=True)
