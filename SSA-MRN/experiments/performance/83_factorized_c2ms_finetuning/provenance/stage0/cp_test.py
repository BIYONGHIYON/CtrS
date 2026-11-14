"""D6 (training-free): is a trained dense 3x3 W2 convolution closer to a bottleneck (3x3 c->r, 1x1 r->c) or to a depthwise-separable chain
(1x1 c->R, depthwise 3x3, 1x1 R->c = CP decomposition of the kernel tensor)? Weight-space factorizations of the 53 C2MS W2 branch (QB seed 42),
then RR test without fine-tuning. Both are compared at similar multiply-accumulate cost."""
import sys, json, copy, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
spec = importlib.util.spec_from_file_location('m79', PERF / '79_deep_narrow_branch_projection/model.py'); M79 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M79)
from common.metrics import rr_metrics
torch.set_grad_enabled(False); torch.set_num_threads(6); DEV = torch.device('mps'); S = 2047.

def svd_conv(conv, r):
    W = conv.weight.double().reshape(conv.out_channels, -1); U, Sv, Vh = torch.linalg.svd(W, full_matrices=False)
    c1 = nn.Conv2d(conv.in_channels, r, 3, padding=1, bias=False); c2 = nn.Conv2d(r, conv.out_channels, 1)
    c1.weight.copy_(Vh[:r].reshape(r, conv.in_channels, 3, 3).float()); c2.weight.copy_((U[:, :r] * Sv[:r]).reshape(-1, r, 1, 1).float()); c2.bias.copy_(conv.bias)
    return nn.Sequential(c1, c2), float(1 - (Sv[:r] ** 2).sum() / (Sv ** 2).sum())

def cp_conv(conv, R, iters=150):
    T = conv.weight.double().reshape(conv.out_channels, conv.in_channels, 9); I, J, K = T.shape; g = torch.Generator().manual_seed(0)
    A = torch.randn(I, R, generator=g, dtype=torch.float64) / R ** .5; B = torch.randn(J, R, generator=g, dtype=torch.float64) / R ** .5; C = torch.randn(K, R, generator=g, dtype=torch.float64)
    T1 = T.reshape(I, J * K); T2 = T.permute(1, 0, 2).reshape(J, I * K); T3 = T.permute(2, 0, 1).reshape(K, I * J); eye = torch.eye(R, dtype=torch.float64)
    kr = lambda X, Y: (X[:, None, :] * Y[None, :, :]).reshape(-1, R)            # Khatri-Rao (rows of X major)
    for _ in range(iters):
        G = (B.T @ B) * (C.T @ C); A = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T1 @ kr(B, C)).T).T
        G = (A.T @ A) * (C.T @ C); B = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T2 @ kr(A, C)).T).T
        G = (A.T @ A) * (B.T @ B); C = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T3 @ kr(A, B)).T).T
    err = float(((torch.einsum('ir,jr,kr->ijk', A, B, C) - T) ** 2).sum() / (T ** 2).sum())
    c1 = nn.Conv2d(J, R, 1, bias=False); dw = nn.Conv2d(R, R, 3, padding=1, groups=R, bias=False); c2 = nn.Conv2d(R, I, 1)
    c1.weight.copy_(B.T.reshape(R, J, 1, 1).float()); dw.weight.copy_(C.T.reshape(R, 1, 3, 3).float()); c2.weight.copy_(A.reshape(I, R, 1, 1).float()); c2.bias.copy_(conv.bias)
    return nn.Sequential(c1, dw, c2), err

def evaluate(model, rr):
    model = model.to(DEV).eval(); rows = []
    for x in rr:
        t = [torch.from_numpy((a / S).astype('float32'))[None].to(DEV) for a in (x['pan'], x['lms'], x['ms'])]
        r = rr_metrics(x['gt'], model(*t).cpu().double()[0].numpy() * S); rows.append([float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')])
    return np.mean(rows, 0)

with h5py.File(ROOT / 'data/test/test_qb_multiExm1.h5') as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
base = p66.load_model(p66.models(42, 'QB')['C2MS'], 4); b = evaluate(copy.deepcopy(base), rr); sh = [(1, 1, 256, 256), (1, 4, 256, 256), (1, 4, 64, 64)]; g0 = M79.conv_macs(copy.deepcopy(base).cpu(), sh) / 1e9
print('base', np.round(b, 4), f'GMAC {g0:.2f}', flush=True); res = {'base': b.tolist()}
for kind, arg in (('svd', 96), ('svd', 64), ('cp', 512), ('cp', 384), ('cp', 256), ('cp', 192)):
    m = copy.deepcopy(base).cpu(); errs = []; t0 = time.time()
    for blk in m.core.backbone.branch:
        if hasattr(blk, 'net'):
            for i in (0, 2):
                new, e = (svd_conv(blk.net[i], arg) if kind == 'svd' else cp_conv(blk.net[i], arg)); blk.net[i] = new; errs.append(e)
    a = evaluate(m, rr); gm = M79.conv_macs(copy.deepcopy(m).cpu(), sh) / 1e9
    res[f'{kind}{arg}'] = dict(test=a.tolist(), gmac=gm, weight_rel_err=float(np.mean(errs)))
    print(f'W2 {kind} {arg:3d}: GMAC {gm:6.2f} ({gm / g0 * 100:3.0f}%)  weight rel err {np.mean(errs):.4f}  test dPSNR {a[0] - b[0]:+.4f} dSAM {a[1] - b[1]:+.4f}  {time.time() - t0:.0f}s', flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
