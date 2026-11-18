"""84 data helpers (the 58 trainer imports this file as `prepare`): experiment 83 prepare.py re-exported unchanged (the experiment 54 sensor
table, Wald operators, 2x coarser pairs and the K10 observation-consistency projection of C2MS-SP). No side effects at import."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp83_prepare', Path(__file__).resolve().parent.parent / '83_factorized_c2ms_finetuning/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected, PROJECTION_ITERATIONS = _m.make_D, _m.Projector, _m.Projected, _m.PROJECTION_ITERATIONS
