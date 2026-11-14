"""Parameter / conv-MAC table of candidate backbones (no training). MACs at 256x256, counted like 79 conv_macs (every Conv2d)."""
import sys, importlib.util, time
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('m79', PERF / '79_deep_narrow_branch_projection/model.py'); M79 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M79)
torch.set_grad_enabled(False)

class Plain(nn.Module):                       # C2MS block: 3x3 - ReLU - 3x3 + skip
    def __init__(s, c): super().__init__(); s.n = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))
    def forward(s, x): return x + s.n(x)
class MB(nn.Module):                          # inverted residual: 1x1 expand - DW3x3 - act - 1x1 project (no norm)
    def __init__(s, c, e=2): super().__init__(); m = int(c * e); s.n = nn.Sequential(nn.Conv2d(c, m, 1), nn.Conv2d(m, m, 3, padding=1, groups=m), nn.GELU(), nn.Conv2d(m, c, 1))
    def forward(s, x): return x + s.n(x)
class SimpleGate(nn.Module):
    def forward(s, x): a, b = x.chunk(2, 1); return a * b
class LN2d(nn.Module):                        # per-pixel channel LayerNorm (NAFNet)
    def __init__(s, c): super().__init__(); s.w = nn.Parameter(torch.ones(1, c, 1, 1)); s.b = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(s, x): u = x.mean(1, keepdim=True); v = (x - u).pow(2).mean(1, keepdim=True); return (x - u) / (v + 1e-6).sqrt() * s.w + s.b
class NAF(nn.Module):                         # NAFNet block without the global SCA pooling
    def __init__(s, c):
        super().__init__(); s.a = nn.Sequential(LN2d(c), nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c), SimpleGate(), nn.Conv2d(c, c, 1))
        s.b = nn.Sequential(LN2d(c), nn.Conv2d(c, 2 * c, 1), SimpleGate(), nn.Conv2d(c, c, 1)); s.be = nn.Parameter(torch.zeros(1, c, 1, 1)); s.ga = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(s, x): y = x + s.be * s.a(x); return y + s.ga * s.b(y)
class PC(nn.Module):                          # FasterNet: partial 3x3 on c/4 channels - 1x1 expand - act - 1x1
    def __init__(s, c, e=2): super().__init__(); s.k = c // 4; s.p = nn.Conv2d(s.k, s.k, 3, padding=1); s.n = nn.Sequential(nn.Conv2d(c, int(c * e), 1), nn.GELU(), nn.Conv2d(int(c * e), c, 1))
    def forward(s, x): a, b = x[:, :s.k], x[:, s.k:]; return x + s.n(torch.cat([s.p(a), b], 1))
class Half(nn.Module):                        # 3x3 on half the channels (dense), 1x1 mix: GPU-friendly "partial dense" block
    def __init__(s, c): super().__init__(); s.k = c // 2; s.p = nn.Sequential(nn.Conv2d(s.k, s.k, 3, padding=1), nn.ReLU(True), nn.Conv2d(s.k, s.k, 3, padding=1)); s.m = nn.Conv2d(c, c, 1)
    def forward(s, x): a, b = x[:, :s.k], x[:, s.k:]; return x + s.m(torch.cat([s.p(a), b], 1))
BLOCK = dict(plain=Plain, mb=MB, naf=NAF, pc=PC, half=Half)

class TwoRes(nn.Module):
    """stem 3x3 (cin -> cf) -> [full-res: nf blocks @cf] + [half-res: unshuffle(2) -> 1x1 -> nh blocks @ch -> 1x1 -> shuffle(2)] -> 3x3 (cf -> bands); y = LMS + detail."""
    def __init__(s, bands, cin, cf, nf, bf, ch, nh, bh):
        super().__init__(); s.stem = nn.Conv2d(cin, cf, 3, padding=1); s.full = nn.Sequential(*[BLOCK[bf](cf) for _ in range(nf)])
        s.low = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * cf, ch, 1), *[BLOCK[bh](ch) for _ in range(nh)], nn.Conv2d(ch, 4 * cf, 1), nn.PixelShuffle(2)) if nh else None
        s.head = nn.Conv2d(cf, bands, 3, padding=1)
    def forward(s, x, lms):
        f = s.stem(x); f = s.full(f) + (s.low(f) if s.low is not None else 0); return lms + s.head(f)

def macs(m, *shapes):
    tot = [0]
    def hook(mod, _i, o): tot[0] += o.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
    hs = [x.register_forward_hook(hook) for x in m.modules() if isinstance(x, nn.Conv2d)]
    m.eval()(*[torch.zeros(s) for s in shapes]); [h.remove() for h in hs]; return tot[0]
def timeit(m, *shapes, n=5):
    return 0.0  # timing disabled: CPU timing on this machine was not reliable (only parameters and conv MACs are reported)
    xs = [torch.zeros(s) for s in shapes]; m.eval()(*xs); t = []
    for _ in range(n): t0 = time.perf_counter(); m(*xs); t.append(time.perf_counter() - t0)
    return sorted(t)[len(t) // 2] * 1e3

rows = []
for bands in (4, 8):
    c2 = M79.M54.Model('C2MS', bands, 42) if hasattr(M79.M54, 'Model') else None
    sh = [(1, 1, 256, 256), (1, bands, 256, 256), (1, bands, 64, 64)]
    if c2 is not None:
        rows.append((f'C2MS (bands {bands})', sum(p.numel() for p in c2.parameters()), M79.used_parameters(c2), macs(c2, *sh), timeit(c2, *sh)))
    for name, cfg in {
        'plain 64x4 + plain 128x8 (C2MS topology, no SSAI)': dict(cf=64, nf=4, bf='plain', ch=128, nh=8, bh='plain'),
        'plain 64x4 + MB 128x8': dict(cf=64, nf=4, bf='plain', ch=128, nh=8, bh='mb'),
        'plain 64x4 + MB 128x12': dict(cf=64, nf=4, bf='plain', ch=128, nh=12, bh='mb'),
        'plain 64x4 + MB 192x8': dict(cf=64, nf=4, bf='plain', ch=192, nh=8, bh='mb'),
        'plain 64x4 + NAF 128x8': dict(cf=64, nf=4, bf='plain', ch=128, nh=8, bh='naf'),
        'plain 64x4 + PC 128x8': dict(cf=64, nf=4, bf='plain', ch=128, nh=8, bh='pc'),
        'plain 64x4 + half 128x8': dict(cf=64, nf=4, bf='plain', ch=128, nh=8, bh='half'),
        'MB 64x4 + MB 128x8': dict(cf=64, nf=4, bf='mb', ch=128, nh=8, bh='mb'),
        'MB 64x8 + MB 128x8': dict(cf=64, nf=8, bf='mb', ch=128, nh=8, bh='mb'),
        'MB 64x8 + MB 128x12': dict(cf=64, nf=8, bf='mb', ch=128, nh=12, bh='mb'),
        'MB 64x8 + MB 192x12': dict(cf=64, nf=8, bf='mb', ch=192, nh=12, bh='mb'),
        'NAF 64x8 + NAF 128x12': dict(cf=64, nf=8, bf='naf', ch=128, nh=12, bh='naf'),
        'plain 48x4 + MB 128x12': dict(cf=48, nf=4, bf='plain', ch=128, nh=12, bh='mb'),
        'plain 32x4 + MB 128x12': dict(cf=32, nf=4, bf='plain', ch=128, nh=12, bh='mb'),
        'half 64x4 + MB 128x12': dict(cf=64, nf=4, bf='half', ch=128, nh=12, bh='mb'),
    }.items():
        m = TwoRes(bands, 2 * bands, **cfg); p = sum(q.numel() for q in m.parameters())
        rows.append((f'{name} (bands {bands})', p, p, macs(m, (1, 2 * bands, 256, 256), (1, bands, 256, 256)), timeit(m, (1, 2 * bands, 256, 256), (1, bands, 256, 256))))
print(f"{'model':62s} {'params':>10s} {'used':>10s} {'GMAC@256':>9s} {'CPU ms':>8s}")
for r in rows: print(f'{r[0]:62s} {r[1]:10,d} {r[2]:10,d} {r[3] / 1e9:9.2f} {r[4]:8.0f}')
