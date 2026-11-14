"""83 arms: the trained 65 C2MS_L1S with its 3x3 convolutions factorized, then fine-tuned briefly with the 65 loss (train.py).

FT0 : 65 C2MS_L1S unchanged (control: the same fine-tuning without factorization)
K1  : the 8 convolutions (64 -> 64, 3x3) of the original full-resolution backbone -> 3x3 64->32 (no bias) + 1x1 32->64, initialized by an
      activation-aware truncated SVD (second moment of the unfolded 3x3 inputs over 512 training patches)
K2  : K1 + the 16 convolutions (128 -> 128, 3x3) of the W2 branch -> 1x1 128->384 (no bias), depthwise 3x3 (384, no bias), 1x1 384->128,
      initialized by a rank-384 CP decomposition of each kernel (alternating least squares, 150 iterations, seed 0, balanced factors)
K3  : K1 + the W2 convolutions at CP rank 256
K2L : K2 fine-tuned for 40 epochs instead of 10 (schedule-length check)
Model(arm, bands, seed) builds the structure only (C2MS of experiment 54, then the factorized modules); the trained initial weights are made
once by `run.py factorize` (factorize() below) and loaded by the trainer. The SSA stages, the decoder and the rest are untouched."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp79_model', HERE.parent / '79_deep_narrow_branch_projection/model.py')
M79 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M79)
M54, M43 = M79.M54, M79.M43
conv_macs, used_parameters = M79.conv_macs, M79.used_parameters
SHAPES = {'FT0': (None, None), 'K1': (32, None), 'K2': (32, 384), 'K3': (32, 256), 'K2L': (32, 384)}  # (backbone rank, W2 CP rank)
ARMS = tuple(SHAPES)
CAL_PATCHES, CAL_SEED, CP_ITERS = 512, 0, 150


def targets(model):
    """(name, parent, key, group) for the 3x3 convolutions of the full-resolution backbone ('bb') and the W2 branch ('w2')."""
    out = []; par = model.core.backbone
    for i, blk in enumerate(par.original):
        out += [(f'bb{i}.conv20', blk, 'conv20', 'bb'), (f'bb{i}.conv21', blk, 'conv21', 'bb')]
    for i, blk in enumerate(par.branch):
        if hasattr(blk, 'net'): out += [(f'w2_{i}.0', blk.net, 0, 'w2'), (f'w2_{i}.2', blk.net, 2, 'w2')]
    return out


def _get(parent, key): return getattr(parent, key) if isinstance(key, str) else parent[key]
def _put(parent, key, mod):
    if isinstance(key, str): setattr(parent, key, mod)
    else: parent[key] = mod


def bottleneck(c, r): return nn.Sequential(nn.Conv2d(c, r, 3, padding=1, bias=False), nn.Conv2d(r, c, 1))
def separable(c, r): return nn.Sequential(nn.Conv2d(c, r, 1, bias=False), nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False), nn.Conv2d(r, c, 1))


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    m = M54.Model('C2MS', bands, seed); rb, rw = SHAPES[arm]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _n, parent, key, g in targets(m):
            c = _get(parent, key).out_channels
            if g == 'bb' and rb: _put(parent, key, bottleneck(c, rb))
            if g == 'w2' and rw: _put(parent, key, separable(c, rw))
    return m


@torch.no_grad()
def second_moments(model, batches, device):
    """Uncentered second moment of the unfolded 3x3 inputs of every backbone convolution (float64, on the CPU)."""
    model = model.to(device).eval(); acc, cnt, hooks = {}, {}, []
    def mk(name):
        def hook(_m, inp, _o):
            x = F.unfold(inp[0].float(), 3, padding=1).transpose(1, 2).reshape(-1, inp[0].shape[1] * 9)
            g = (x.T @ x).cpu().double(); acc[name] = acc[name] + g if name in acc else g; cnt[name] = cnt.get(name, 0) + x.shape[0]
        return hook
    for n, parent, key, g in targets(model):
        if g == 'bb': hooks.append(_get(parent, key).register_forward_hook(mk(n)))
    for b in batches: model(*[t.to(device) for t in b])
    for h in hooks: h.remove()
    return {n: acc[n] / cnt[n] for n in acc}


def svd_factor(conv, sigma, r):
    """Activation-aware truncated SVD: minimizes E||W x - A B x||^2 over the calibration inputs (eigendecomposition, robust to dead inputs)."""
    W = conv.weight.detach().double().cpu().reshape(conv.out_channels, -1)
    lam, Q = torch.linalg.eigh(sigma); lam = lam.clamp_min(lam.max() * 1e-9)
    U, S, Vh = torch.linalg.svd(W @ (Q * lam.sqrt()), full_matrices=False)
    A = U[:, :r] * S[:r]; B = Vh[:r] @ (Q / lam.sqrt()).T
    new = bottleneck(conv.in_channels, r)
    new[0].weight.copy_(B.reshape(r, conv.in_channels, 3, 3).float()); new[1].weight.copy_(A.reshape(-1, r, 1, 1).float()); new[1].bias.copy_(conv.bias.detach().cpu())
    return new, float(1 - (S[:r] ** 2).sum() / (S ** 2).sum())


def cp_factor(conv, R, iters=CP_ITERS, seed=CAL_SEED):
    """CP decomposition T[o, i, k] ~ sum_r A[o, r] B[i, r] C[k, r] by ALS -> 1x1 (B), depthwise 3x3 (C), 1x1 (A, conv bias)."""
    T = conv.weight.detach().double().cpu().reshape(conv.out_channels, conv.in_channels, 9); I, J, K = T.shape; g = torch.Generator().manual_seed(seed)
    A = torch.randn(I, R, generator=g, dtype=torch.float64) / R ** .5; B = torch.randn(J, R, generator=g, dtype=torch.float64) / R ** .5; C = torch.randn(K, R, generator=g, dtype=torch.float64)
    T1 = T.reshape(I, J * K); T2 = T.permute(1, 0, 2).reshape(J, I * K); T3 = T.permute(2, 0, 1).reshape(K, I * J); eye = torch.eye(R, dtype=torch.float64)
    kr = lambda X, Y: (X[:, None, :] * Y[None, :, :]).reshape(-1, R)
    for _ in range(iters):
        G = (B.T @ B) * (C.T @ C); A = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T1 @ kr(B, C)).T).T
        G = (A.T @ A) * (C.T @ C); B = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T2 @ kr(A, C)).T).T
        G = (A.T @ A) * (B.T @ B); C = torch.linalg.solve(G + 1e-9 * G.diagonal().mean() * eye, (T3 @ kr(A, B)).T).T
    na, nb, nc = A.norm(dim=0), B.norm(dim=0), C.norm(dim=0); s = (na * nb * nc).clamp_min(1e-30) ** (1 / 3)
    A, B, C = A / na.clamp_min(1e-30) * s, B / nb.clamp_min(1e-30) * s, C / nc.clamp_min(1e-30) * s
    err = float(((torch.einsum('ir,jr,kr->ijk', A, B, C) - T) ** 2).sum() / (T ** 2).sum())
    new = separable(J, R)
    new[0].weight.copy_(B.T.reshape(R, J, 1, 1).float()); new[1].weight.copy_(C.T.reshape(R, 1, 3, 3).float()); new[2].weight.copy_(A.reshape(I, R, 1, 1).float()); new[2].bias.copy_(conv.bias.detach().cpu())
    return new, err


@torch.no_grad()
def factorize(base, arm, sigma):
    """Trained 65 model (M54 C2MS) -> initial weights of `arm` (same parameter names as Model(arm)); returns (model, report)."""
    m = copy.deepcopy(base).cpu(); rb, rw = SHAPES[arm]; report = {'bb_energy_lost': [], 'w2_relative_error': []}
    for n, parent, key, g in targets(m):
        conv = _get(parent, key)
        if g == 'bb' and rb:
            new, e = svd_factor(conv, sigma[n], rb); _put(parent, key, new); report['bb_energy_lost'].append(e)
        if g == 'w2' and rw:
            new, e = cp_factor(conv, rw); _put(parent, key, new); report['w2_relative_error'].append(e)
    report = {k: (float(np.mean(v)) if v else None) for k, v in report.items()}
    return m, report
