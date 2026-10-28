"""67 arms: unchanged C2MS architecture (54 Model('C2MS')); the arms differ only in the training loss and the PAN input augmentation (train.py)."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
# C2MS_*: MSE (C2MS loss); L1S_*: L1 + 0.46 x spectral loss (65 loss). _G: PAN brightness, _B: PAN blur, _GB: both.
ARMS=('C2MS_G','C2MS_B','C2MS_GB','L1S_G','L1S_B','L1S_GB')
CORE43={'C2MS':'C2',**{a:'C2' for a in ARMS}}
def Model(arm,bands=4,seed=42):
    if arm not in ('C2MS',)+ARMS:raise ValueError(arm)
    return M54.Model('C2MS',bands,seed)
