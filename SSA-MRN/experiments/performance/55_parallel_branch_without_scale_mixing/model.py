"""55 arm C2: original SSA-MRN K4 + W2 parallel branch (43 C2) trained on original pairs only.

Built exactly like 54 Model('C2MS') of the same seed and band count, so C2 and the 54 C2MS start from identical weights and
differ only in the 2x coarser pair loss (coarse_weight 0 here). The 54 C0 and C2MS weights are the comparison arms."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C2',)
CORE43 = {'C2': 'C2'}


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M54.Model('C2MS', bands, seed)
