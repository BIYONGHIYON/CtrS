"""65 arm C2MS_L1S: unchanged C2MS architecture (54 Model('C2MS')), trained with L1 + the 57/58 spectral loss (train.py)."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
ARMS=('C2MS_L1S',)
CORE43={'C2MS':'C2','C2MS_L1S':'C2'}
def Model(arm,bands=4,seed=42):
    if arm not in ('C2MS',)+ARMS:raise ValueError(arm)
    return M54.Model('C2MS',bands,seed)
