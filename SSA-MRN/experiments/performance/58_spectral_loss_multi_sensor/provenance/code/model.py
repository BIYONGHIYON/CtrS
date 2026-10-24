"""58 arm C2MS_SAM: the C2MS baseline (original SSA-MRN K4 + W2 parallel branch, scale-mixed training) + the 57 spectral loss.

Builds exactly 54 Model('C2MS') for the band count of the sensor, so on 4-band sensors it equals 53 C2MS / 43 C2 of the same seed
and on 8-band WV3 it equals 54 C2MS. The controls are the 53 (QB) and 54 (GF2, WV3) C2MS runs of the same seeds."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp54_model', HERE.parent / '54_multi_sensor_scale_mixed_training/model.py')
M54 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M54)
M43 = M54.M43
ARMS = ('C2MS_SAM',)
CORE43 = {'C2MS_SAM': 'C2'}


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M54.Model('C2MS', bands, seed)
