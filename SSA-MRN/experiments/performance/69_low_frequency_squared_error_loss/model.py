"""69 arms: unchanged C2MS architecture (54 Model('C2MS')); the arms differ only in the reconstruction loss (train.py)."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
# C2MS_L1SO: 65 loss + low-band observation term; C2MS_SPLIT: squared error on the low band, L1 on the high band, + 65 spectral term.
ARMS=('C2MS_L1SO','C2MS_SPLIT')
CORE43={'C2MS':'C2',**{a:'C2' for a in ARMS}}
def Model(arm,bands=4,seed=42):
    if arm not in ('C2MS',)+ARMS:raise ValueError(arm)
    return M54.Model('C2MS',bands,seed)
