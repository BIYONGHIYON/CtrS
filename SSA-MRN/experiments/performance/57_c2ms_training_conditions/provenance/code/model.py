"""57 arms: the C2MS baseline (original SSA-MRN K4 + W2 parallel branch, 53) under three training conditions, all with an EMA of the weights.

Every arm builds exactly 54 Model('C2MS'), which on 4-band QB equals 53 C2MS / 43 C2 of the same seed, so all arms and the 53 C2MS
control start from identical weights. They differ only in the schedule length and the loss (config.json epochs, spectral_weight)."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C2MS', 'C2MS_L300', 'C2MS_SAM')
CORE43 = {arm: 'C2' for arm in ARMS}


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M54.Model('C2MS', bands, seed)
