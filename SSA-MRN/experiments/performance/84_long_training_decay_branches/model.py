"""84: the unchanged C2MS architecture of C2MS-SP (54 Model('C2MS')) for every run; only the training length and schedule differ (train.py).

TRUNK   constant learning rate 1e-4 for 800 epochs, full training state saved at epochs 120, 240, 480 and 800
B<N>    branch from the TRUNK state at 0.8 N, then 0.2 N epochs of cosine decay 1e-4 -> 1e-6 (N = 150, 300, 600, 1000 epochs in total)"""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
ARMS=('TRUNK','B150','B300','B600','B1000')
BRANCHES={'B150':150,'B300':300,'B600':600,'B1000':1000}
CORE43={'C2MS':'C2',**{a:'C2' for a in ARMS}}
def Model(arm,bands=4,seed=42):
    if arm not in ('C2MS',)+ARMS:raise ValueError(arm)
    return M54.Model('C2MS',bands,seed)
