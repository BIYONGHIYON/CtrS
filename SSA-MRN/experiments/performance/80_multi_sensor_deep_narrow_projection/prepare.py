"""80 data helpers and the inference projection: experiment 79 prepare.py re-exported unchanged (no side effects at import; the 58 trainer
imports this file as `prepare`).

79 re-exports the experiment 54 sensor table (QB, GF2, WV3 and WV2 for the transfer test), Wald operators and 2x coarser pairs, and holds
the observation-consistency projection of 45/66 (Projector, Projected): y' = y + D^T M^T (M D D^T M^T)^-1 M (MS - D y), interior mask
(margin 5), conjugate gradient per band in float64 (tolerance 1e-8, at most 400 iterations), with D the Wald operator of the sensor the
image comes from (WV2 operator on the WV2 transfer test)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp79_prepare', Path(__file__).resolve().parent.parent / '79_deep_narrow_branch_projection/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected = _m.make_D, _m.Projector, _m.Projected
