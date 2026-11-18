"""87 data helpers (the 58 trainer imports this file as `prepare`): experiment 86 prepare.py re-exported (the 85/83/79 sensor table, Wald
operators, coarse pairs, the 79/83 K10 projection), with `FastProjected` = the 86 FastProjectedV2 (the K10 projection on the interior grid with
D^T by FFT; equal to the 79 projection up to rounding, checked by 86 on every test scene). No side effects at import."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp86_prepare', Path(__file__).resolve().parent.parent / '86_fft_adjoint_fast_inference/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN, RATIO = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN, _m.RATIO
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector, Projected, PROJECTION_ITERATIONS = _m.make_D, _m.Projector, _m.Projected, _m.PROJECTION_ITERATIONS
FastProjector, FastProjected = _m.FastProjectorV2, _m.FastProjectedV2
