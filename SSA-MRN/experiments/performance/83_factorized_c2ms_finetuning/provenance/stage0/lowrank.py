"""D4 (training-free): data-aware low-rank factorization of the trained 3x3 convolutions of C2MS / C2MS_L1S (QB), no fine-tuning.
Each 3x3 conv (c -> c) becomes 3x3 (c -> r, no bias) + 1x1 (r -> c, bias); the rank-r factors minimize the output error on
activations collected from QB training patches (original + 2x coarser are not mixed: original-scale patches only)."""
import sys, json, copy, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
spec = importlib.util.spec_from_file_location('m79', PERF / '79_deep_narrow_branch_projection/model.py'); M79 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M79)
from common.metrics import rr_metrics
torch.set_grad_enabled(False); torch.set_num_threads(6)
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); S = 2047.

def targets(model):
    """(name, parent module, attribute/index, group) for every 3x3 conv of the full-resolution backbone ('bb') and the W2 branch ('w2')."""
    out = []; par = model.core.backbone
    for i, blk in enumerate(par.original):
        out += [(f'bb{i}.conv20', blk, 'conv20', 'bb'), (f'bb{i}.conv21', blk, 'conv21', 'bb')]
    for i, blk in enumerate(par.branch):
        if hasattr(blk, 'net'): out += [(f'w2_{i}.0', blk.net, 0, 'w2'), (f'w2_{i}.2', blk.net, 2, 'w2')]
    return out
def get(parent, key): return getattr(parent, key) if isinstance(key, str) else parent[key]
def put(parent, key, mod):
    if isinstance(key, str): setattr(parent, key, mod)
    else: parent[key] = mod

def collect(model, batches):
    """Uncentered second moment of the unfolded 3x3 inputs of every target conv."""
    model = model.to(DEV).eval(); tg = targets(model); acc = {n: None for n, *_ in tg}; cnt = {n: 0 for n, *_ in tg}; hooks = []
    def mk(name):
        def hook(mod, inp, _out):
            x = F.unfold(inp[0].float(), 3, padding=1).transpose(1, 2).reshape(-1, inp[0].shape[1] * 9)
            g = (x.T @ x).cpu().double(); acc[name] = g if acc[name] is None else acc[name] + g; cnt[name] += x.shape[0]
        return hook
    for n, parent, key, _g in tg: hooks.append(get(parent, key).register_forward_hook(mk(n)))
    for b in batches: model(*[t.to(DEV) for t in b])
    for h in hooks: h.remove()
    return {n: acc[n] / cnt[n] for n in acc}

def factorize(conv, sigma, r):
    W = conv.weight.detach().double().cpu().reshape(conv.out_channels, -1); d = W.shape[1]
    lam, Q = torch.linalg.eigh(sigma); lam = lam.clamp_min(lam.max() * 1e-9)          # sigma = (Q sqrt(lam)) (Q sqrt(lam))^T, robust to dead channels
    U, Sv, Vh = torch.linalg.svd(W @ (Q * lam.sqrt()), full_matrices=False)
    A = U[:, :r] * Sv[:r]; B = Vh[:r] @ (Q / lam.sqrt()).T
    c1 = nn.Conv2d(conv.in_channels, r, 3, padding=1, bias=False); c2 = nn.Conv2d(r, conv.out_channels, 1, bias=True)
    c1.weight.copy_(B.reshape(r, conv.in_channels, 3, 3).float()); c2.weight.copy_(A.reshape(conv.out_channels, r, 1, 1).float()); c2.bias.copy_(conv.bias.detach().cpu())
    energy = float((Sv[:r] ** 2).sum() / (Sv ** 2).sum())
    return nn.Sequential(c1, c2), energy

def compress(model, sig, ranks):
    m = copy.deepcopy(model).cpu(); kept = []
    for n, parent, key, g in targets(m):
        r = ranks[g]; conv = get(parent, key)
        if r >= conv.out_channels: continue
        new, e = factorize(conv, sig[n], r); put(parent, key, new); kept.append(e)
    return m, (float(np.mean(kept)) if kept else 1.0)

def evaluate(model, rr, proj=False, K=10):
    model = model.to(DEV).eval(); rows = []
    for x in rr:
        t = [torch.from_numpy((a / S).astype('float32'))[None].to(DEV) for a in (x['pan'], x['lms'], x['ms'])]
        y = model(*t).cpu().double()
        if proj: y, _ = p66.project(y, torch.from_numpy(x['ms'] / S)[None], max_iter=K)
        r = rr_metrics(x['gt'], y[0].numpy() * S); rows.append([float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')])
    return np.mean(rows, 0)

if __name__ == '__main__':
    out = Path(sys.argv[1]); seed = int(sys.argv[2]); n_cal = int(sys.argv[3]); res = {}
    with h5py.File(ROOT / 'data/train/train_qb.h5') as f:
        idx = np.sort(np.random.default_rng(0).choice(len(f['pan']), n_cal, replace=False))
        cal = {k: torch.from_numpy(np.asarray(f[k][idx], dtype=np.float32) / S) for k in ('pan', 'lms', 'ms')}
    batches = [tuple(cal[k][i:i + 32] for k in ('pan', 'lms', 'ms')) for i in range(0, n_cal, 32)]
    with h5py.File(ROOT / 'data/test/test_qb_multiExm1.h5') as f:
        rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
    with h5py.File(ROOT / 'data/validation/valid_qb.h5') as f:
        vi = np.linspace(0, len(f['pan']) - 1, 400).astype(int); val = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in vi]
    p66.use('QB')
    for name, path in p66.models(seed, 'QB').items():
        if name not in ('C2MS', 'L1S'): continue
        base = p66.load_model(path, 4); t0 = time.time(); sig = collect(copy.deepcopy(base), batches)
        sh = [(1, 1, 256, 256), (1, 4, 256, 256), (1, 4, 64, 64)]
        b = evaluate(copy.deepcopy(base), rr); bv = evaluate(copy.deepcopy(base), val)
        res[name] = {'base': dict(test=b.tolist(), val=bv.tolist(), gmac=M79.conv_macs(copy.deepcopy(base).cpu(), sh) / 1e9, params=M79.used_parameters(copy.deepcopy(base).cpu()))}
        print(name, 'base test', np.round(b, 4), 'val', np.round(bv, 4), f"GMAC {res[name]['base']['gmac']:.2f} params {res[name]['base']['params']:,}", f'{time.time() - t0:.0f}s', flush=True)
        for w2, bb in ((96, 64), (64, 64), (48, 64), (32, 64), (128, 48), (128, 32), (128, 24), (96, 48), (64, 32), (48, 32), (48, 24), (32, 24), (32, 16)):
            m, e = compress(base, sig, dict(w2=w2, bb=bb)); a = evaluate(m, rr); av = evaluate(m, val)
            g = M79.conv_macs(copy.deepcopy(m).cpu(), sh) / 1e9; p = M79.used_parameters(copy.deepcopy(m).cpu())
            res[name][f'w2r{w2}_bbr{bb}'] = dict(test=a.tolist(), val=av.tolist(), gmac=g, params=p, kept_energy=e)
            print(f'{name} w2 rank {w2:3d} bb rank {bb:2d}  GMAC {g:6.2f} ({g / res[name]["base"]["gmac"] * 100:4.0f}%)  params {p:9,d}  test dPSNR {a[0] - b[0]:+.4f} dSAM {a[1] - b[1]:+.4f} dERGAS {a[2] - b[2]:+.4f}   val dPSNR {av[0] - bv[0]:+.4f}', f'{time.time() - t0:.0f}s', flush=True)
            out.write_text(json.dumps(res, indent=1))
