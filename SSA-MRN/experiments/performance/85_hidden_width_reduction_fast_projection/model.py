"""85 arms: the trained 65 C2MS_L1S with the middle (hidden) channels of its residual blocks reduced, then fine-tuned with the 65 loss.

FT0L   : 65 C2MS_L1S unchanged (control: the same 40-epoch fine-tuning without reduction)
B64W96 : W2 branch blocks 128 -> 96 -> 128 (backbone unchanged)
B32W96 : full-resolution backbone blocks 64 -> 32 -> 64 and W2 blocks 128 -> 96 -> 128
B32W64 : backbone blocks 64 -> 32 -> 64 and W2 blocks 128 -> 64 -> 128
Only plain 3x3 convolutions with fewer channels; no layer is added. Kept channels: largest contribution energy E[h_c^2] * ||W_b[:, c]||^2 of
the ReLU output h over CAL_PATCHES training patches; the first 3x3 keeps its rows, the second 3x3 (weights and bias) is refit by ridge least
squares to the original block's residual output. Blocks are reduced one at a time in forward order with the inputs of the already-reduced
model. Model(arm, bands, seed) builds the structure only; reduce() makes the initial weights (run.py stage0). The SSA stages, the decoder,
the W2 PixelUnshuffle/1x1/PixelShuffle and the input path are untouched."""
import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp83_model', HERE.parent / '83_factorized_c2ms_finetuning/model.py')
M83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M83)
M54, conv_macs, used_parameters = M83.M54, M83.conv_macs, M83.used_parameters
WIDTHS = {'FT0L': (64, 128), 'B64W96': (64, 96), 'B32W96': (32, 96), 'B32W64': (32, 64)}  # (backbone middle, W2 middle)
ARMS = tuple(WIDTHS)
CAL_PATCHES, CAL_SEED, RIDGE = 512, 0, 1e-6


def blocks(model):
    """(name, parent, first key, second key, group) in forward order; parent's input is the block input."""
    par = model.core.backbone; out = []
    for i, b in enumerate(par.original): out.append((f'bb{i}', b, 'conv20', 'conv21', 'bb'))
    for i, b in enumerate(par.branch):
        if hasattr(b, 'net'): out.append((f'w2_{i}', b.net, 0, 2, 'w2'))
    return out


def Model(arm, bands=4, seed=42):
    if arm not in ARMS: raise ValueError(arm)
    m = M54.Model('C2MS', bands, seed); hb, hw = WIDTHS[arm]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _n, parent, k1, k2, g in blocks(m):
            h = hb if g == 'bb' else hw; a, b = M83._get(parent, k1), M83._get(parent, k2)
            if h < a.out_channels:
                M83._put(parent, k1, nn.Conv2d(a.in_channels, h, 3, padding=1)); M83._put(parent, k2, nn.Conv2d(h, b.out_channels, 3, padding=1))
    return m


class _Stop(Exception):
    pass


@torch.no_grad()
def _inputs(model, parent, batches, device):
    """Inputs of `parent` for every calibration batch (the forward stops there)."""
    xs = []
    def hook(_m, inp):
        xs.append(inp[0].detach().clone()); raise _Stop
    h = parent.register_forward_pre_hook(hook)
    try:
        for b in batches:
            try: model(*[t.to(device) for t in b])
            except _Stop: pass
    finally: h.remove()
    return xs


@torch.no_grad()
def reduce(base, arm, batches, device):
    """Trained 65 model (M54 C2MS) -> initial weights of `arm` (same parameter names as Model(arm)); returns (model on CPU, report)."""
    m = copy.deepcopy(base).to(device).eval(); hb, hw = WIDTHS[arm]; report = {}
    for name, parent, k1, k2, g in blocks(m):
        h = hb if g == 'bb' else hw; a, b = M83._get(parent, k1), M83._get(parent, k2)
        if h >= a.out_channels: continue
        xs = _inputs(m, parent, batches, device); co = b.out_channels
        energy = torch.zeros(a.out_channels, dtype=torch.float64, device=device); count = 0
        for x in xs:
            hid = F.relu(a(x)); energy += hid.double().square().sum((0, 2, 3)); count += hid.shape[0] * hid.shape[2] * hid.shape[3]
        score = energy / count * b.weight.double().square().sum((0, 2, 3))
        keep = torch.sort(torch.topk(score, h).indices).values
        dim = h * 9 + 1; G = torch.zeros(dim, dim, dtype=torch.float64, device=device); C = torch.zeros(dim, co, dtype=torch.float64, device=device)
        for x in xs:
            hid = F.relu(a(x)); t = b(hid).permute(0, 2, 3, 1).reshape(-1, co)
            f = F.unfold(hid[:, keep], 3, padding=1).transpose(1, 2).reshape(-1, h * 9); f = torch.cat([f, torch.ones(len(f), 1, device=device)], 1)
            G += (f.T @ f).double(); C += (f.T @ t).double()
        reg = torch.eye(dim, dtype=torch.float64, device=device) * (RIDGE * G.diagonal()[:-1].mean()); reg[-1, -1] = 0
        W = torch.linalg.solve(G + reg, C)
        na = nn.Conv2d(a.in_channels, h, 3, padding=1).to(device); nb = nn.Conv2d(h, co, 3, padding=1).to(device)
        na.weight.copy_(a.weight[keep]); na.bias.copy_(a.bias[keep])
        nb.weight.copy_(W[:-1].T.reshape(co, h, 3, 3).float()); nb.bias.copy_(W[-1].float())
        err = tot = 0.
        for x in xs:
            t = b(F.relu(a(x))); p = nb(F.relu(na(x))); err += float((p - t).double().square().sum()); tot += float(t.double().square().sum())
        M83._put(parent, k1, na); M83._put(parent, k2, nb)
        report[name] = dict(kept=h, of=a.out_channels, energy_kept=float(score[keep].sum() / score.sum()), relative_residual_error=err / max(tot, 1e-30))
    return m.cpu(), report
