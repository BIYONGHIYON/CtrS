"""80 arm: the 79 C2MS_W64D16_L1S model unchanged (79 model.py), trained on GF2 (4 bands) and WV3 (8 bands).

C2MS_W64D16_L1S: original SSA-MRN K4 + W2 branch PixelUnshuffle(2) -> 1x1 (4*64)->64 -> 16 residual blocks at 64 channels -> 1x1 64->256 ->
                 PixelShuffle(2) (75 SizedParallel). Built like 75/79 Model (core first, then the branch, inside one forked RNG), so on any
                 band count every layer outside the branch equals 54 C0 of the same seed and band count, and the initial output equals it."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp79_model', HERE.parent / '79_deep_narrow_branch_projection/model.py')
M79 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M79)
M75, M54, M43 = M79.M75, M79.M54, M79.M43
conv_macs, used_parameters = M79.conv_macs, M79.used_parameters
ARMS = ('C2MS_W64D16_L1S',)
SHAPES = {a: M79.SHAPES[a] for a in ARMS}


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    return M79.Model(arm, bands, seed)
