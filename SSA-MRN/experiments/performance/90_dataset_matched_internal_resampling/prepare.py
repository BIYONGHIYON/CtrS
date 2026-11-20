"""90 data helpers (the 58 trainer imports this file as `prepare`): experiment 89 prepare.py re-exported (88/86 sensor table, Wald
operators, coarse pairs, `FastProjected` = 86 FastProjectedV2 K10 projection, `Graphed`). No side effects."""
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp89_prepare', _here.parent / '89_spectral_coarse_loss_weights/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN, RATIO = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN, _m.RATIO
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected, PROJECTION_ITERATIONS = _m.make_D, _m.Projector, _m.Projected, _m.PROJECTION_ITERATIONS
FastProjector, FastProjected, Graphed = _m.FastProjector, _m.FastProjected, _m.Graphed
