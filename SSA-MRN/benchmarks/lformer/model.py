"""LFormer adapter: official code in upstream/ (coder-JMHou/LFormer, unmodified); inputs and output in DN / scale.

Model: upstream/models/LFormer.py AttnFuseMain with the official configs/config_<sensor>.yaml sizes
(pan_dim 1, n_stage 5; lms_dim / attn_dim / hp_dim: WV3 8/64/64, QB 4/32/48, GF2 4/64/64). WV2 uses the WV3 network
(transfer). The official network takes (lms, pan), predicts a residual and adds lms; ms is not used. In eval mode the
output is clipped to [0, 1] as the official val_step does; in training mode it is not (official train_step: loss on
the unclipped output).

Inference without PatchMergeModule: the official test script tiles the image with a compiled PatchMergeModule
(Linux .so only, patch_merge: True). Training never uses it (train_step calls _forward_implem directly). The adapter
builds the network with patch_merge=False and runs the whole image at once; window attention (8x8) only needs H and W
divisible by 8 (RR 256, FR 512). compare_patch_merge.py checks the difference against the official tiling on Linux.

Checkpoints: author weights (upstream/weights/WV3/LFormer_epoch500.pth) and ours both store the network under 'model'.
"""
import importlib
import importlib.util
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
PACKAGE = 'lformer_upstream_models'


def official(name='LFormer'):
    """Import upstream/models as its own package (relative imports keep working, no clash with other 'models')."""
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, UPSTREAM / 'models/__init__.py',
                                                      submodule_search_locations=[str(UPSTREAM / 'models')])
        module = importlib.util.module_from_spec(spec); sys.modules[PACKAGE] = module; spec.loader.exec_module(module)
    return importlib.import_module(f'{PACKAGE}.{name}')


# configs/config_<sensor>.yaml (lms_dim, attn_dim, hp_dim); the paper reports no channel sizes
DIMS = {'WV3': (8, 64, 64), 'QB': (4, 32, 48), 'GF2': (4, 64, 64)}
TRAINING_SENSOR = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}
N_STAGE, CROP_BATCH, PATCH_SIZES, SCALE, WINDOW = 5, 64, [16, 64, 64], 4, 8   # config defaults, FirstAttn window_size
AUTHOR_WEIGHTS = UPSTREAM / 'weights/WV3/LFormer_epoch500.pth'
AUTHOR_SHA256 = '8b521a5f8e8bbe8104ebb11f1c97e91d9a0a19d3f994d342cd3952551827170a'


def _checkpointed(module, *args):
    forward = type(module).forward
    return checkpoint(forward, module, *args, use_reentrant=False) if torch.is_grad_enabled() else forward(module, *args)


class LFormer(nn.Module):
    def __init__(self, sensor, patch_merge=False):
        super().__init__()
        lms_dim, attn_dim, hp_dim = DIMS[TRAINING_SENSOR[sensor]]
        self.net = official().AttnFuseMain(1, lms_dim, attn_dim, hp_dim, N_STAGE, patch_merge, CROP_BATCH, PATCH_SIZES, SCALE)
        self.patch_merge = patch_merge

    def set_checkpointing(self, enabled):
        """Recompute each stage in the backward pass (torch.utils.checkpoint, non-reentrant, RNG preserved for the attention
        dropout) instead of storing its activations: same loss and gradients, only more time."""
        mods = [self.net.attn, self.net.pre_hp, *self.net.refined_blocks, *self.net.hp_branch, self.net.final_conv]
        for m in mods:
            if enabled: m.forward = partial(_checkpointed, m)
            else: m.__dict__.pop('forward', None)
        return self

    def forward(self, pan, lms, ms=None):
        if self.training:
            return self.net._forward_implem(lms, pan) + lms                  # official train_step (unclipped)
        if self.patch_merge:
            return self.net.val_step(ms, lms, pan)                           # official tiling (Linux .so only)
        h, w = lms.shape[-2:]
        if h % WINDOW or w % WINDOW: raise ValueError(f'H and W must be multiples of {WINDOW}, got {h}x{w}')
        return (self.net._forward_implem(lms, pan) + lms).clip(0, 1)         # official val_step without tiling


def build(bands, sensor, setting, seed, patch_merge=False):
    net = LFormer(sensor, patch_merge)
    if bands != DIMS[TRAINING_SENSOR[sensor]][0]: raise ValueError(f'{sensor}: {bands} bands')
    return net


def load(net, path):
    """Author weights and our best.pt: {'model': AttnFuseMain state_dict, ...}."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state, strict=True)
    return net
