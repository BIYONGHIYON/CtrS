"""88 arms: C2MS (the C2MS-SP structure, 54 Model('C2MS')) for the trunk and the full-size decay, and the 83 K2 structure (C2MS-SP-Lite) for
the two factorized runs. Factorization is the 83 code (activation-aware SVD of the backbone, CP-ALS of W2).

TRUNK : C2MS, constant lr 1e-4, 120 epochs from the seed initialization (equal to 43 C2 / 65 / 84 on 4-band sensors)
FULL  : C2MS, the TRUNK epoch-120 state + 30 epochs of cosine decay (84 B150 recipe)
K2D   : K2 from the TRUNK epoch-120 weights, 30 epochs of cosine decay 1e-4 -> 1e-6 (official candidate)
K2F   : K2 from the FULL best weights, 10 epochs 5e-5 -> 1e-6 (the 83 Lite recipe; reference)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp83_model', Path(__file__).resolve().parent.parent / '83_factorized_c2ms_finetuning/model.py')
M83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M83)
M54, M43, conv_macs, used_parameters = M83.M54, M83.M43, M83.conv_macs, M83.used_parameters
CAL_PATCHES, CAL_SEED, second_moments, factorize = M83.CAL_PATCHES, M83.CAL_SEED, M83.second_moments, M83.factorize
FULL_SIZE = ('TRUNK', 'FULL')
FACTORIZED = ('K2D', 'K2F')
ARMS = FULL_SIZE + FACTORIZED
CORE43 = {a: 'C2' for a in ('C2MS',) + ARMS}  # the 58 preflight compares 4-band full-size initial weights with 43 C2
SOURCE = {'K2D': 'TRUNK', 'K2F': 'FULL'}  # run whose weights each factorized arm starts from


def Model(arm, bands=4, seed=42):
    if arm in ('C2MS', 'L1S', 'B150') + FULL_SIZE: return M54.Model('C2MS', bands, seed)
    if arm in ('K2',) + FACTORIZED: return M83.Model('K2', bands, seed)
    raise ValueError(arm)
