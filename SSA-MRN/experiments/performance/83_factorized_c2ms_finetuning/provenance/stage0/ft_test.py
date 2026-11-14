"""D7 (local feasibility, not an official training): does a short fine-tune recover the accuracy of a factorized C2MS?
53 C2MS (QB seed 42): full-resolution backbone -> rank-32 bottleneck (data-aware SVD), W2 convs -> 1x1 / depthwise 3x3 / 1x1 (CP rank R).
Fine-tune a few hundred steps on QB training patches with the C2MS loss (original pair MSE + 2x coarser pair MSE), then QB validation / RR test."""
import sys, json, copy, time, math, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
LR = load('lowrank', Path(__file__).parent / 'lowrank.py'); CP = None
p66 = LR.p66; M79 = LR.M79
P54 = load('prep54', PERF / '54_multi_sensor_scale_mixed_training/prepare.py')
from common.metrics import rr_metrics
DEV = torch.device('mps'); S = 2047.; torch.set_num_threads(6)

def cp_conv(conv, R, iters=150):
    T = conv.weight.detach().double().cpu().reshape(conv.out_channels, conv.in_channels, 9); I, J, K = T.shape; g = torch.Generator().manual_seed(0)
    A = torch.randn(I, R, generator=g, dtype=torch.float64) / R ** .5; B = torch.randn(J, R, generator=g, dtype=torch.float64) / R ** .5; C = torch.randn(K, R, generator=g, dtype=torch.float64)
    T1 = T.reshape(I, J * K); T2 = T.permute(1, 0, 2).reshape(J, I * K); T3 = T.permute(2, 0, 1).reshape(K, I * J); eye = torch.eye(R, dtype=torch.float64)
    kr = lambda X, Y: (X[:, None, :] * Y[None, :, :]).reshape(-1, R)
    for _ in range(iters):
        G = (B.T @ B) * (C.T @ C); A = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T1 @ kr(B, C)).T).T
        G = (A.T @ A) * (C.T @ C); B = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T2 @ kr(A, C)).T).T
        G = (A.T @ A) * (B.T @ B); C = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T3 @ kr(A, B)).T).T
    # balance the three factors so that no layer starts with extreme scales
    na, nb, nc = A.norm(dim=0), B.norm(dim=0), C.norm(dim=0); s = (na * nb * nc) ** (1 / 3); A, B, C = A / na * s, B / nb * s, C / nc * s
    c1 = nn.Conv2d(J, R, 1, bias=False); dw = nn.Conv2d(R, R, 3, padding=1, groups=R, bias=False); c2 = nn.Conv2d(R, I, 1)
    with torch.no_grad():
        c1.weight.copy_(B.T.reshape(R, J, 1, 1).float()); dw.weight.copy_(C.T.reshape(R, 1, 3, 3).float()); c2.weight.copy_(A.reshape(I, R, 1, 1).float()); c2.bias.copy_(conv.bias.detach().cpu())
    return nn.Sequential(c1, dw, c2)

@torch.no_grad()
def evaluate(model, sets):
    model = model.to(DEV).eval(); out = []
    for data in sets:
        rows = []
        for x in data:
            t = [torch.from_numpy((a / S).astype('float32'))[None].to(DEV) for a in (x['pan'], x['lms'], x['ms'])]
            r = rr_metrics(x['gt'], model(*t).cpu().double()[0].numpy() * S); rows.append([float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')])
        out.append(np.mean(rows, 0))
    return out

if __name__ == '__main__':
    out = Path(sys.argv[1]); steps = int(sys.argv[2]); configs = [c.split(':') for c in sys.argv[3].split(',')]; lr0 = float(sys.argv[4]); res = {}
    torch.set_grad_enabled(False)
    with h5py.File(ROOT / 'data/train/train_qb.h5') as f:
        idx = np.sort(np.random.default_rng(1).choice(len(f['pan']), 6144, replace=False)); tr = {k: torch.from_numpy(np.asarray(f[k][idx], dtype=np.float32) / S) for k in ('pan', 'lms', 'ms', 'gt')}
    with h5py.File(ROOT / 'data/test/test_qb_multiExm1.h5') as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
    with h5py.File(ROOT / 'data/validation/valid_qb.h5') as f:
        vi = np.linspace(0, len(f['pan']) - 1, 300).astype(int); val = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in vi]
    coarse = P54.CoarsePairs('QB', 'cpu'); sh = [(1, 1, 256, 256), (1, 4, 256, 256), (1, 4, 64, 64)]
    base = p66.load_model(p66.models(42, 'QB')['C2MS'], 4)
    cal = [tuple(tr[k][i:i + 32] for k in ('pan', 'lms', 'ms')) for i in range(0, 512, 32)]; sig = LR.collect(copy.deepcopy(base), cal)
    b = evaluate(copy.deepcopy(base), (rr, val)); g0 = M79.conv_macs(copy.deepcopy(base).cpu(), sh) / 1e9; res['base'] = dict(test=b[0].tolist(), val=b[1].tolist(), gmac=g0)
    print('base test', np.round(b[0], 4), 'val', np.round(b[1], 4), f'GMAC {g0:.2f}', flush=True)
    for bb, R in configs:
        bb, R = int(bb), int(R); name = f'bb{bb}_cp{R}'; t0 = time.time()
        m, _ = LR.compress(base, sig, dict(w2=10 ** 6, bb=bb))                                  # backbone only (data-aware SVD); w2 untouched here
        if R > 0:
            for blk in m.core.backbone.branch:
                if hasattr(blk, 'net'):
                    for i in (0, 2): blk.net[i] = cp_conv(blk.net[i], R)
        gm = M79.conv_macs(copy.deepcopy(m).cpu(), sh) / 1e9; par = M79.used_parameters(copy.deepcopy(m).cpu())
        a = evaluate(copy.deepcopy(m), (rr, val)); log = [dict(step=0, test=a[0].tolist(), val=a[1].tolist())]
        print(f'{name}: GMAC {gm:.2f} ({gm / g0 * 100:.0f}%) params {par:,}  step 0  test dPSNR {a[0][0] - b[0][0]:+.3f} dSAM {a[0][1] - b[0][1]:+.3f}  val dPSNR {a[1][0] - b[1][0]:+.3f}  {time.time() - t0:.0f}s', flush=True)
        torch.set_grad_enabled(True); m = m.to(DEV).train(); opt = torch.optim.Adam(m.parameters(), lr=lr0); g = torch.Generator().manual_seed(0)
        for it in range(1, steps + 1):
            for gr in opt.param_groups: gr['lr'] = 1e-6 + (lr0 - 1e-6) * 0.5 * (1 + math.cos(math.pi * it / steps))
            bi = torch.randint(0, 6144, (16,), generator=g); pan, lms, ms, gt = (tr[k][bi] for k in ('pan', 'lms', 'ms', 'gt'))
            with torch.no_grad(): pc, lc, mc, gc = coarse(pan, gt)
            loss = (m(pan.to(DEV), lms.to(DEV), ms.to(DEV)) - gt.to(DEV)).square().mean() + (m(pc.to(DEV), lc.to(DEV), mc.to(DEV)) - gc.to(DEV)).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if it % 250 == 0 or it == steps:
                torch.set_grad_enabled(False); a = evaluate(copy.deepcopy(m), (rr, val)); torch.set_grad_enabled(True); m.train()
                log.append(dict(step=it, test=a[0].tolist(), val=a[1].tolist()))
                print(f'{name}: step {it:4d}  test dPSNR {a[0][0] - b[0][0]:+.3f} dSAM {a[0][1] - b[0][1]:+.3f} dERGAS {a[0][2] - b[0][2]:+.3f}  val dPSNR {a[1][0] - b[1][0]:+.3f}  {time.time() - t0:.0f}s', flush=True)
        torch.set_grad_enabled(False); res[name] = dict(gmac=gm, params=par, log=log); out.write_text(json.dumps(res, indent=1))
