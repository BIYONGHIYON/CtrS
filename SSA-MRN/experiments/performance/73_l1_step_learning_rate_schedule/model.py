"""Unchanged C2MS architecture of 53/54 for both arms; only the training loss and the learning-rate schedule change (train.py)."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
ARMS=('C2MS_L1S','C2MS_L1C')
CORE43={arm:'C2' for arm in ARMS}
def Model(arm,bands=4,seed=42):
    if arm not in ARMS:raise ValueError(arm)
    return M54.Model('C2MS',bands,seed)
