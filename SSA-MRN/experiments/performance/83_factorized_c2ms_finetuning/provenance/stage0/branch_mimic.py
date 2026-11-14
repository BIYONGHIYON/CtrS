"""D5 (local proxy, not an official training): can a cheaper W2 branch reproduce the trained C2MS W2 branch?
Everything except the new branch is the frozen 53 C2MS (QB seed 42). The student branch regresses the teacher branch output (feature MSE)
on QB training patches for a fixed number of steps; it is then plugged into C2MS and evaluated on QB validation patches and RR test
without any end-to-end fine-tuning. plain 64x8 / 64x16 are calibration points with known from-scratch results (75, 79)."""
import sys, json, copy, time, math, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
spec = importlib.util.spec_from_file_location('m79', PERF / '79_deep_narrow_branch_projection/model.py'); M79 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M79)
from common.metrics import rr_metrics
torch.set_num_threads(4); torch.set_grad_enabled(True); DEV = torch.device('mps'); S = 2047.

class Plain(nn.Module):
    def __init__(s, c): super().__init__(); s.n = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))
    def forward(s, x): return x + s.n(x)
class MB(nn.Module):                          # inverted residual, no normalization
    def __init__(s, c, e=2): super().__init__(); m = int(c * e); s.n = nn.Sequential(nn.Conv2d(c, m, 1), nn.Conv2d(m, m, 3, padding=1, groups=m), nn.GELU(), nn.Conv2d(m, c, 1))
    def forward(s, x): return x + s.n(x)
class MB2(nn.Module):                         # two depthwise 3x3 per block (same spatial depth as a plain block)
    def __init__(s, c, e=2): super().__init__(); m = int(c * e); s.n = nn.Sequential(nn.Conv2d(c, m, 1), nn.Conv2d(m, m, 3, padding=1, groups=m), nn.GELU(), nn.Conv2d(m, m, 3, padding=1, groups=m), nn.GELU(), nn.Conv2d(m, c, 1))
    def forward(s, x): return x + s.n(x)
class GP(nn.Module):                          # grouped dense 3x3 (groups 4) with a 1x1 mix
    def __init__(s, c, g=4): super().__init__(); s.n = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=g), nn.ReLU(True), nn.Conv2d(c, c, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1, groups=g))
    def forward(s, x): return x + s.n(x)
BL = dict(plain=Plain, mb=MB, mb2=MB2, gp=GP)
def branch(kind, w, d, stem=64):
    b = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * stem, w, 1), *[BL[kind](w) for _ in range(d)], nn.Conv2d(w, 4 * stem, 1), nn.PixelShuffle(2))
    nn.init.zeros_(b[-2].weight); nn.init.zeros_(b[-2].bias); return b

def stem(core, pan, lms):
    pu = core.prelu(core.conv1t1(core.upsample1(core.downsample1(pan)))); return core.cov2t64(torch.cat([lms, pan, pu], 1))

@torch.no_grad()
def evaluate(model, rr, val):
    model = model.to(DEV).eval(); out = []
    for data in (rr, val):
        rows = []
        for x in data:
            t = [torch.from_numpy((a / S).astype('float32'))[None].to(DEV) for a in (x['pan'], x['lms'], x['ms'])]
            r = rr_metrics(x['gt'], model(*t).cpu().double()[0].numpy() * S); rows.append([float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')])
        out.append(np.mean(rows, 0))
    return out

if __name__ == '__main__':
    out = Path(sys.argv[1]); steps = int(sys.argv[2]); cands = [c.split(':') for c in sys.argv[3].split(',')]; res = json.loads(out.read_text()) if out.exists() else {}
    g = torch.Generator().manual_seed(0)
    with h5py.File(ROOT / 'data/train/train_qb.h5') as f:
        idx = np.sort(np.random.default_rng(1).choice(len(f['pan']), 4096, replace=False)); tr = {k: torch.from_numpy(np.asarray(f[k][idx], dtype=np.float32) / S) for k in ('pan', 'lms')}
    with h5py.File(ROOT / 'data/test/test_qb_multiExm1.h5') as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
    with h5py.File(ROOT / 'data/validation/valid_qb.h5') as f:
        vi = np.linspace(0, len(f['pan']) - 1, 300).astype(int); val = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in vi]
    base = p66.load_model(p66.models(42, 'QB')['C2MS'], 4).to(DEV).eval()
    for p in base.parameters(): p.requires_grad_(False)
    core = base.core; teacher = core.backbone.branch; sh = [(1, 1, 256, 256), (1, 4, 256, 256), (1, 4, 64, 64)]
    if 'base' not in res:
        b = evaluate(base, rr, val); res['base'] = dict(test=b[0].tolist(), val=b[1].tolist(), gmac=M79.conv_macs(copy.deepcopy(base).cpu(), sh) / 1e9); base.to(DEV)
    print('base', res['base'], flush=True)
    for kind, w, d in cands:
        w, d = int(w), int(d); name = f'{kind}_{w}x{d}'
        if name in res: print('skip', name); continue
        torch.manual_seed(0); st = branch(kind, w, d).to(DEV); opt = torch.optim.Adam(st.parameters(), lr=3e-4); t0 = time.time(); ema = None
        for it in range(steps):
            for gr in opt.param_groups: gr['lr'] = 1e-5 + (3e-4 - 1e-5) * 0.5 * (1 + math.cos(math.pi * it / steps))
            b = torch.randint(0, 4096, (16,), generator=g); pan, lms = tr['pan'][b].to(DEV), tr['lms'][b].to(DEV)
            with torch.no_grad(): f = stem(core, pan, lms); tgt = teacher(f)
            loss = F.mse_loss(st(f), tgt) / tgt.pow(2).mean().clamp_min(1e-12)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
            if it % 200 == 0 or it == steps - 1: print(f'  {name} step {it:5d} rel feature mse {ema:.4f}  {time.time() - t0:.0f}s', flush=True)
        m = copy.deepcopy(base); m.core.backbone.branch = st.eval()
        a = evaluate(m, rr, val); gm = M79.conv_macs(copy.deepcopy(m).cpu(), sh) / 1e9; par = sum(p.numel() for p in st.parameters())
        res[name] = dict(test=a[0].tolist(), val=a[1].tolist(), gmac=gm, branch_params=par, rel_feature_mse=ema, steps=steps, seconds=time.time() - t0)
        print(f"{name:14s} GMAC {gm:6.2f} ({gm / res['base']['gmac'] * 100:3.0f}%) branch params {par:9,d}  rel mse {ema:.4f}  test dPSNR {a[0][0] - res['base']['test'][0]:+.3f} dSAM {a[0][1] - res['base']['test'][1]:+.3f}  val dPSNR {a[1][0] - res['base']['val'][0]:+.3f}  {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res, indent=1)); base.to(DEV)
