"""Single C2MS model; dilation metadata differs only for experiment 70 C2MS_D."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
ARMS=('BLUR_BOTH', 'BLUR_COARSE')
CORE43={a:"C2" for a in ("C2MS",)+ARMS}
def Model(arm,bands=4,seed=42):
    if arm not in ("C2MS",)+ARMS:raise ValueError(arm)
    m=M54.Model("C2MS",bands,seed)
    return m
