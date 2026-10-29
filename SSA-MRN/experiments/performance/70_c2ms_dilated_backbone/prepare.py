"""Sensor table, Wald operators and 2x coarser pairs of experiment 54, used unchanged (the 53 construction, as in 56 and 65)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp54_prepare', Path(__file__).resolve().parent.parent / '54_multi_sensor_scale_mixed_training/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE = _m.SENSORS, _m.PHASE, _m.SIZE
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
