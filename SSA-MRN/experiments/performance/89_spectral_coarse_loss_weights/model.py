"""89 arms. Every trained arm uses the W2 2x2 implementation of the C2MS W2 branch (to_2x2):

  input : PixelUnshuffle(2) -> Conv2d(256, 128, 1)  ==  Conv2d(64, 128, 2, stride 2); the 2x2 weight is the seeded 1x1 weight reshaped
          (PixelUnshuffle channel c*4 + 2i + j -> input channel c, kernel position (i, j)), so the initial function is unchanged
  output: Conv2d(128, 256, 1) -> PixelShuffle(2)  ->  2x2 stride-2 transposed convolution 128 -> 64 with a per-channel bias (64 instead of
          256 per-position biases), zero initialized like the original; implemented either as ConvTranspose2d ('convT') or as
          Conv2d(128, 256, 1, bias=False) -> PixelShuffle(2) -> per-channel bias ('pixelshuffle'), mathematically identical (OUT_IMPL)

W2T          : C2MS (2x2), lr 1e-4 constant, 120 epochs (the 88 TRUNK recipe)
W2K          : 83 K2 structure (2x2) from the W2T epoch-120 weights, 30-epoch cosine decay (the 88 K2D recipe); the W2 2x2 check
L<lam>C<c>   : C2MS (2x2) screening run (stage B: BL<lam>C<c>), 50 epochs (40 at 1e-4 + 10 cosine), loss weights lambda = lam, c = c
Other names are passed to the 88 model (controls: 'C2MS'/'L1S' = 65 C2MS-SP weights, 'K2D' = 88 K2D, 'K2' = 83 Lite)."""
import importlib.util
import re
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp88_model', HERE.parent / '88_factorization_at_decay_start/model.py')
M88 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M88)
M83, M54, M43 = M88.M83, M88.M54, M88.M43
conv_macs, used_parameters, CAL_PATCHES, CAL_SEED, second_moments, factorize = M88.conv_macs, M88.used_parameters, M88.CAL_PATCHES, M88.CAL_SEED, M88.second_moments, M88.factorize
SCREEN = re.compile(r'^B?L(\d+\.\d\d)C(\d+\.\d\d)$')  # stage-B arms carry the prefix B (the (lambda*, 1.0) reference is retrained)
FULL_SIZE = ('W2T',)
FACTORIZED = ('W2K',)
SOURCE = {'W2K': 'W2T'}
REFERENCE_ARM = 'L0.46C1.00'
ARMS = ('W2T', 'W2K', REFERENCE_ARM)  # structures exercised by the 58 preflight (every screening arm has the W2T structure)
CORE43 = {a: 'C2' for a in ARMS}
OUT_IMPL = {'value': 'convT'}  # set from the sealed runtime config by train.py / evaluate.py / run.py


def screen_arm(lam, c, stage='A'): return ('B' if stage == 'B' else '') + f'L{lam:.2f}C{c:.2f}'


def parse(arm):
    """(lambda, c) of a trained arm."""
    m = SCREEN.match(arm)
    if m: return float(m.group(1)), float(m.group(2))
    if arm in ('W2T', 'W2K'): return 0.46, 1.0
    raise ValueError(arm)


def is_screen(arm): return bool(SCREEN.match(arm))


class PSOut(nn.Module):
    """Conv2d(128, 4*out, 1, bias=False) -> PixelShuffle(2) -> per-channel bias; equal to ConvTranspose2d(128, out, 2, 2)."""

    def __init__(self, cin, cout):
        super().__init__(); self.conv = nn.Conv2d(cin, 4 * cout, 1, bias=False); self.shuffle = nn.PixelShuffle(2); self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x): return self.shuffle(self.conv(x)) + self.bias.view(1, -1, 1, 1)


def out_layer(impl, cin=128, cout=64):
    if impl == 'convT': m = nn.ConvTranspose2d(cin, cout, 2, stride=2)
    elif impl == 'pixelshuffle': m = PSOut(cin, cout)
    else: raise ValueError(impl)
    for p in m.parameters(): nn.init.zeros_(p)
    return m


def set_out_weight(layer, w1):
    """Load a 1x1 weight w1 [4*cout, cin] (PixelShuffle order) into either output implementation (tests only)."""
    cout = w1.shape[0] // 4
    if isinstance(layer, nn.ConvTranspose2d): layer.weight.copy_(w1.reshape(cout, 2, 2, -1).permute(3, 0, 1, 2))
    else: layer.conv.weight.copy_(w1.reshape(4 * cout, -1, 1, 1))


@torch.no_grad()
def to_2x2(m, impl=None):
    """Replace the W2 branch input (PixelUnshuffle + 1x1) by the equal 2x2 stride-2 convolution and the output (1x1 + PixelShuffle) by the
    2x2 transposed convolution. Consumes no global RNG."""
    impl = impl or OUT_IMPL['value']; par = m.core.backbone; br = par.branch
    if not (isinstance(br[0], nn.PixelUnshuffle) and isinstance(br[-1], nn.PixelShuffle)): raise ValueError('unexpected W2 branch')
    w_in = br[1]; cin = w_in.in_channels // 4
    with torch.random.fork_rng(devices=[]):
        new_in = nn.Conv2d(cin, w_in.out_channels, 2, stride=2); out = out_layer(impl, br[-2].in_channels, br[-2].out_channels // 4)
    new_in.weight.copy_(w_in.weight.reshape(w_in.out_channels, cin, 2, 2)); new_in.bias.copy_(w_in.bias)
    par.branch = nn.Sequential(new_in, *list(br)[2:-2], out)
    return m


def Model(arm, bands=4, seed=42):
    if arm == 'W2T' or is_screen(arm): return to_2x2(M54.Model('C2MS', bands, seed))
    if arm == 'W2K': return to_2x2(M83.Model('K2', bands, seed))
    return M88.Model(arm, bands, seed)
