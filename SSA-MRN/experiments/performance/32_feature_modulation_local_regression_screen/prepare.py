"""Phase 0 (no training, validation only): D3 error split, D4 GF complementarity, D5 FR proxy."""
import os, sys, json, time, argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import numpy as np, torch
from torch.nn import functional as F
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
from common.metrics import fr_metrics
import importlib
guided_prior = importlib.import_module('experiments.performance.32_feature_modulation_local_regression_screen.model').guided_prior

EPS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def lowpass(x, h, w):
    d = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
    return F.interpolate(d, size=x.shape[-2:], mode='bicubic', align_corners=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a0', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    suite.configure(42, 'cuda')
    ck = torch.load(a.a0, map_location='cpu', weights_only=False)
    net = RestoredPansharpeningNet(4, 4); net.load_state_dict(ck['model'], strict=True); net.cuda().eval()
    va = suite.Cached(os.path.join(a.cache, 'val')); n = len(va)
    pan, lms, ms, gt = (torch.from_numpy(np.array(va.arrays[k])) for k in ('pan', 'lms', 'ms', 'gt'))
    pred = []
    with torch.inference_mode():
        for i in range(0, n, 64):
            pred.append(net(pan[i:i+64].cuda(), lms[i:i+64].cuda(), ms[i:i+64].cuda()).cpu())
    pred = torch.cat(pred).double(); gtd = gt.double(); lmsd = lms.double()
    base_mse = float(((pred - gtd) ** 2).mean())
    # D3: error energy split at the MS scale (low = MS-representable band)
    e = (gtd - pred).float()
    elow = torch.cat([lowpass(e[i:i+256].cuda(), 16, 16).cpu() for i in range(0, n, 256)]).double()
    tot = float((e.double() ** 2).sum())
    d3 = {'a0_val_mse': base_mse, 'low_energy_share': float((elow ** 2).sum()) / tot, 'high_energy_share': float(((e.double() - elow) ** 2).sum()) / tot,
          'per_band_mse': [float(((pred[:, b] - gtd[:, b]) ** 2).mean()) for b in range(4)], 'lms_val_mse': float(((lmsd - gtd) ** 2).mean())}
    half = n // 2
    A, B = slice(0, half), slice(half, n)

    def blend_fit(gf):
        coef, mse_b = [], 0.
        for b in range(4):
            X = torch.stack([pred[A, b].reshape(-1), gf[A, b].reshape(-1), torch.ones(half * 4096, dtype=torch.float64)], 1)
            y = gtd[A, b].reshape(-1)
            w = torch.linalg.solve(X.T @ X, X.T @ y)
            coef.append(w.tolist())
        return coef

    def blend_apply(gf, coef, idx):
        return torch.stack([coef[b][0] * pred[idx, b] + coef[b][1] * gf[idx, b] + coef[b][2] for b in range(4)], 1)

    d4 = {'eps': {}}
    best = None
    for eps in EPS:
        gf = torch.cat([guided_prior(pan[i:i+256].cuda(), ms[i:i+256].cuda(), eps).cpu() for i in range(0, n, 256)]).double()
        coef = blend_fit(gf)
        a_mse_A = float(((blend_apply(gf, coef, A) - gtd[A]) ** 2).mean())
        row = {'gf_alone_mse': float(((gf - gtd) ** 2).mean()), 'blend_mse_halfA': a_mse_A}
        d4['eps'][str(eps)] = row
        if best is None or a_mse_A < best[1]:
            best = (eps, a_mse_A, gf, coef)
    eps, _, gf, coef = best
    a0_B = float(((pred[B] - gtd[B]) ** 2).mean())
    bl_B = float(((blend_apply(gf, coef, B) - gtd[B]) ** 2).mean())
    d4.update(selected_eps=eps, selection='blend MSE on validation first half only', coef_per_band=coef, a0_mse_halfB=a0_B, blend_mse_halfB=bl_B,
              relative_reduction_percent_halfB=100 * (a0_B - bl_B) / a0_B)
    # D5: FR-style proxy (Dlambda, Ds, QNR) on 256 fixed validation patches, reduced scale
    idx = list(range(0, n, n // 256))[:256]
    blend_all = blend_apply(gf, coef, slice(0, n))
    d5 = {}
    for name, img in (('LMS', lmsd), ('A0', pred), ('GF', gf), ('blend', blend_all)):
        rows = [fr_metrics(img[i].numpy() * 2047, lms[i].double().numpy() * 2047, ms[i].double().numpy() * 2047, pan[i, 0].double().numpy() * 2047) for i in idx]
        d5[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    gate = d4['relative_reduction_percent_halfB'] >= 2 or (d5['A0']['D_s'] - d5['blend']['D_s'] >= 0.005 and bl_B <= a0_B)
    out = {'D3': d3, 'D4': d4, 'D5_val_fr_proxy_256': d5, 'w2_gate_passed': bool(gate), 'rule': 'D4 >= 2% or (blend Ds better by >=0.005 with MSE not worse)', 'time': time.time()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
