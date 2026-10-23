"""56 arm C0MS: original SSA-MRN K4 (no W2 branch) trained on original + 2x coarser pairs (the 53 scale-mixed training).

Built exactly like 54 Model('C0') of the same seed, which on 4-band QB equals 43 C0, so C0MS and the 43 C0 control start from
identical weights and differ only in the coarse pair loss. With 43 C0/C2 and 53 C2MS this completes the 2x2 of W2 x scale mixing."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C0MS',)
CORE43 = {'C0MS': 'C0'}


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M54.Model('C0', bands, seed)
