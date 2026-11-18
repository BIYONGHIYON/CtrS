"""87 arms: the 85 reduced structures (plain 3x3 residual blocks with fewer middle channels, channel selection + least-squares refit of the
second 3x3 from the trained 65 C2MS_L1S) fine-tuned for 20 epochs with the 65 loss, seed 42 only. Structure and reduction are the 85 code.

B32W96 : backbone blocks 64 -> 32 -> 64, W2 blocks 128 -> 96 -> 128 (40.30 GMAC, 68%; 1,997,946 used parameters, 73%)
B32W64 : backbone blocks 64 -> 32 -> 64, W2 blocks 128 -> 64 -> 128 (30.64 GMAC, 51%; 1,407,866, 51%)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp85_model', Path(__file__).resolve().parent.parent / '85_hidden_width_reduction_fast_projection/model.py')
M85 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M85)
M54, M83, conv_macs, used_parameters, CAL_PATCHES, CAL_SEED = M85.M54, M85.M83, M85.conv_macs, M85.used_parameters, M85.CAL_PATCHES, M85.CAL_SEED
WIDTHS = {a: M85.WIDTHS[a] for a in ('B32W96', 'B32W64')}
ARMS = tuple(WIDTHS)


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M85.Model(arm, bands, seed)


def reduce(base, arm, batches, device):
    if arm not in ARMS: raise ValueError(arm)
    return M85.reduce(base, arm, batches, device)
