"""83 data helpers (the 58 trainer imports this file as `prepare`): experiment 79 prepare.py re-exported unchanged (the experiment 54 sensor
table for QB, GF2, WV3 and the WV2 transfer test, Wald operators, 2x coarser pairs and the observation-consistency projection of 45/66).

The projection used here stops the conjugate gradient after 10 iterations (K10). Experiment 81 measured K10 against the full projection
(tolerance 1e-8, up to 400 iterations) on QB, GF2, WV3 and WV2, 3 seeds: RR PSNR within 0.0003 dB, the same v2.4 judgements, about 6 times
less projection time. No side effects at import."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location('exp79_prepare', Path(__file__).resolve().parent.parent / '79_deep_narrow_branch_projection/prepare.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SENSORS, PHASE, SIZE, OBS_MARGIN = _m.SENSORS, _m.PHASE, _m.SIZE, _m.OBS_MARGIN
data_files, gen_mtf, Wald, CoarsePairs = _m.data_files, _m.gen_mtf, _m.Wald, _m.CoarsePairs
make_D, Projector = _m.make_D, _m.Projector
PROJECTION_ITERATIONS = 10


class Projected(torch.nn.Module):
    """Normalized (pan, lms, ms) in, projected normalized output out (float64), conjugate gradient stopped at K10 (81)."""

    def __init__(self, net, sensor, max_iter=PROJECTION_ITERATIONS):
        super().__init__(); self.net = net; self.projector = Projector(sensor, max_iter=max_iter); self.iterations = []

    def forward(self, pan, lms, ms):
        y = self.net(pan, lms, ms).double()
        with torch.inference_mode(False):
            out, it = self.projector(y.clone(), ms.double().clone())
        self.iterations.append(it)
        return out.detach()
