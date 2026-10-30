"""Single C2MS model; dilation metadata differs only for experiment 70 C2MS_D."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c2ms54_model',HERE.parent/'54_multi_sensor_scale_mixed_training/model.py')
M54=importlib.util.module_from_spec(spec);spec.loader.exec_module(M54)
M43=M54.M43
ARMS=('C2MS_R', 'C2MS_D')
CORE43={a:"C2" for a in ("C2MS",)+ARMS}
def Model(arm,bands=4,seed=42):
    if arm not in ("C2MS",)+ARMS:raise ValueError(arm)
    m=M54.Model("C2MS",bands,seed)
    if arm=="C2MS_D":
        for i in (1,3):
            c=m.core.backbone.original[i].conv21
            c.dilation=(2,2);c.padding=(2,2)
            assert c is getattr(m.core,"res"+str(i+1)).conv21
    return m
