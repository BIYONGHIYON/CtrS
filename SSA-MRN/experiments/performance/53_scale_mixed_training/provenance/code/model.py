"""52 arms: the 48 model (C2 = 43 original+W2, F1 = original+G5 refiners) trained on mixed scales.

C2MS and F1MS build exactly Model('C2', seed) and Model('F1', seed) of experiment 48, so the initial weights equal
the 43 C2 / 49 F1 controls of the same seed. Only the training data differ (see train.py, coarse pairs)."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp48_model', HERE.parent / '48_parallel_capacity_observation_refinement/model.py')
M48 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M48)
ARMS = ('C2MS', 'F1MS')
BASE = {'C2MS': 'C2', 'F1MS': 'F1'}
Parallel = M48.Parallel


def Model(arm, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M48.Model(BASE[arm], seed)
