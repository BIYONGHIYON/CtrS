"""Experiment 72 (no training): split the gain of C2MS over the original SSA-MRN by frequency band and by structure size.

Frequency: FFT of the error per scene and band. Parseval makes the radial bins an exact partition of the MSE, and the LMDI split
           (logarithmic mean) turns it into dB contributions that add up exactly to the PSNR change. SAM is split by an exact
           Shapley value over six frequency groups.
Structure: pixel classes built from the GT intensity only, see classify(). A pixel partition, so the LMDI split is exact too.

Parts: --part check (synthetic structure patterns), predict (test outputs into --cache, checked against the archived scene PSNR),
frequency, structure, canny (the rejected first classes), rr2 (QB, frequency only), swap (direct band swaps as a cross check),
images, summary (aggregates, results/summary.json and the figures). Default: all of them. Writes only below results/ and --cache.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse, importlib.util, itertools, json, math, sys, tempfile
from pathlib import Path
import h5py, numpy as np, torch
from scipy import ndimage as ndi
from skimage.feature import canny

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; PERF = ROOT / 'experiments/performance'
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics  # noqa: E402  (imported before the 43 provenance path is added)
from common import result_images  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


P68 = _module('exp68_prepare', PERF / '68_ssai_folding_sam_split_seed_ensemble/prepare.py')  # FWald, predict and the 54 model
M54, SENSORS, sha = P68.M54, P68.SENSORS, P68.sha
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
SEEDS = tuple(CFG['seeds']); TARGETS = tuple(CFG['targets'])
EDGES = np.array(CFG['frequency']['edges_cycles_per_pixel']); BINS = CFG['frequency']['bins']
GROUPS = [['DC'], ['>64px', '32-64px'], ['16-32px'], ['8-16px'], ['4-8px'], ['2-4px', '<2px(diag)']]
GROUP_NAMES = CFG['frequency']['sam_shapley_groups']
S = CFG['structure']; SCALES = tuple(S['scales']); NAMES = S['classes']; TAU, GATE, COH = S['extent_tau'], S['gate_percentile'], S['coherence']['threshold']
K = 10 / math.log(10); CACHE = None


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding='utf-8')


def read_test(target):
    path = ROOT / f"data/test/test_{SENSORS[target]['tag']}_multiExm1.h5"
    if sha(path) != CFG['data_sha256'][target]: raise ValueError(f'data sha mismatch {path}')
    with h5py.File(path, 'r') as f:
        return {k: np.asarray(f[k], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}


def arms_of(target): return ('C0', 'C2', 'C2MS') if target == 'QB' else ('C0', 'C2MS')
def pairs_of(target): return (('C0', 'C2'), ('C2', 'C2MS'), ('C0', 'C2MS')) if target == 'QB' else (('C0', 'C2MS'),)


def load_net(target, arm, seed):
    sensor = CFG['targets'][target]
    path = ROOT / CFG['checkpoints'][sensor][arm].replace('<seed>', str(seed))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = M54.Model('C0' if arm == 'C0' else 'C2MS', SENSORS[sensor]['bands'], seed); net.load_state_dict(ck['model'], strict=True)
    return net.eval()


def cached(target, arm, seed): return np.load(CACHE / f'{target}_{arm}_s{seed}.npy', mmap_mode='r').astype(np.float64)


# ---------------------------------------------------------------- part: predict

def run_predict(log):
    res = []
    for target in TARGETS:
        data = read_test(target); scale = SENSORS[CFG['targets'][target]]['max_value']; n = len(data['gt'])
        for arm in arms_of(target):
            for seed in SEEDS:
                net = load_net(target, arm, seed)
                ys = np.stack([P68.predict(net, (data['pan'][i], data['lms'][i], data['ms'][i]), scale) for i in range(n)])
                np.save(CACHE / f'{target}_{arm}_s{seed}.npy', ys.astype(np.float32))
                psnr = [rr_metrics(data['gt'][i], ys[i])['PSNR'] for i in range(n)]
                arch = json.load(open(ROOT / CFG['archived_rr'][target][arm].replace('<seed>', str(seed))))['rows']
                diff = float(max(abs(p - r['metrics']['PSNR']) for p, r in zip(psnr, arch)))
                res.append(dict(target=target, arm=arm, seed=seed, mean_psnr=float(np.mean(psnr)), max_abs_diff_vs_archived=diff))
                log(res[-1])
                if diff > CFG['reproduction']['scene_psnr_tolerance_db']: raise ValueError(f'not reproduced {res[-1]}')
    write('reproduction.json', res)


# ---------------------------------------------------------------- frequency

def bin_masks(h, w):
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij'))
    masks = {'DC': r == 0}
    for k, name in enumerate(BINS[1:]): masks[name] = (r >= EDGES[k]) & (r < EDGES[k + 1]) & (r > 0)
    if sum(m.sum() for m in masks.values()) != h * w: raise ValueError('bins are not a partition')
    return masks


def log_mean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (a - b) / (np.log(a) - np.log(b))
    return np.where(np.isclose(a, b), a, out)


def boot(x, n=10000, seed=0):
    """Scene bootstrap of a (scene, item) array of seed means."""
    idx = np.random.default_rng(seed).integers(0, x.shape[0], (n, x.shape[0])); b = x[idx].mean(1)
    return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def spectra(gt, y, masks, window=None):
    """(scene, band, bin) error energy and the terms of y = H g + n inside each bin."""
    w = 1.0 if window is None else window.astype(np.float32)
    G = np.fft.fft2(gt.astype(np.float32) * w, axes=(-2, -1)); Y = np.fft.fft2(y.astype(np.float32) * w, axes=(-2, -1))
    norm = float((gt.shape[-1] * gt.shape[-2]) ** 2) * (1.0 if window is None else float((window ** 2).mean()))
    out = {k: np.zeros(gt.shape[:2] + (len(BINS),)) for k in ('err', 'gg', 'yy', 'cross')}
    for j, name in enumerate(BINS):
        m = masks[name]
        if not m.any(): continue
        Gm, Ym = G[..., m], Y[..., m]
        out['err'][..., j] = (np.abs(Ym - Gm) ** 2).sum(-1, dtype=np.float64) / norm
        out['gg'][..., j] = (np.abs(Gm) ** 2).sum(-1, dtype=np.float64) / norm
        out['yy'][..., j] = (np.abs(Ym) ** 2).sum(-1, dtype=np.float64) / norm
        out['cross'][..., j] = (Ym * np.conj(Gm)).real.sum(-1, dtype=np.float64) / norm
    return out


def frequency_block(target, gt, lms, arms, pairs, masks, hann=True):
    scene = {arm: [spectra(gt, cached(target, arm, s), masks) for s in SEEDS] for arm in arms}
    R = dict(LMS=dict(mse_by_bin=spectra(gt, lms, masks)['err'].mean((0, 1)).tolist()), GT_power=scene[arms[0]][0]['gg'].mean((0, 1)).tolist())
    for arm in arms:
        d = {k: np.stack([x[k] for x in scene[arm]]) for k in scene[arm][0]}
        e, g, xy, yy = d['err'], d['gg'], d['cross'], d['yy']; gsum, ysum, xsum = g.sum((0, 1, 2)), yy.sum((0, 1, 2)), xy.sum((0, 1, 2))
        H = xy / np.maximum(g, 1e-12)
        R[arm] = dict(mse_by_bin=e.mean((0, 1, 2)).tolist(), rmse_total=float(np.sqrt(e.sum(-1).mean())), power_ratio=(ysum / np.maximum(gsum, 1e-12)).tolist(),
                      gain=(xsum / np.maximum(gsum, 1e-12)).tolist(), coherence=(xsum / np.maximum(np.sqrt(ysum * gsum), 1e-12)).tolist(),
                      gain_error=((1 - H) ** 2 * g).mean((0, 1, 2)).tolist(),
                      uncorrelated=np.maximum(yy - xy ** 2 / np.maximum(g, 1e-12), 0).mean((0, 1, 2)).tolist())
    P = {}
    for a, b in pairs:
        C = []
        for k in range(len(SEEDS)):
            ea, eb = scene[a][k]['err'], scene[b][k]['err']
            C.append(K * ((ea - eb) / log_mean(ea.sum(-1), eb.sum(-1))[..., None]).mean(1))  # (scene, bin), mean over bands
        C = np.stack(C); lo, hi = boot(C.mean(0))
        ea = np.stack([x['err'] for x in scene[a]]); eb = np.stack([x['err'] for x in scene[b]])
        P[f'{a}->{b}'] = dict(total_db=float(C.sum(-1).mean()), total_db_seed=C.sum(-1).mean(1).tolist(), db_by_bin=C.mean((0, 1)).tolist(),
                              db_by_bin_seed=C.mean(1).tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(), share=(C.mean((0, 1)) / C.sum(-1).mean()).tolist(),
                              reduction_pct=(100 * (1 - eb.mean((0, 1, 2)) / np.maximum(ea.mean((0, 1, 2)), 1e-12))).tolist(),
                              scenes_improved_by_bin=(C.mean(0) > 0).sum(0).tolist())
    R['pairs'] = P
    if hann:  # the partition ignores that the error is not periodic; a Hann window shows whether that moved the shares
        window = np.outer(np.hanning(gt.shape[-2]), np.hanning(gt.shape[-1])); R['hann_share'] = {}
        for a, b in pairs:
            V = []
            for s in SEEDS:
                ea = spectra(gt, cached(target, a, s), masks, window)['err']; eb = spectra(gt, cached(target, b, s), masks, window)['err']
                V.append(K * ((ea - eb) / log_mean(ea.sum(-1), eb.sum(-1))[..., None]).mean(1))
            V = np.stack(V); R['hann_share'][f'{a}->{b}'] = (V.mean((0, 1)) / V.sum(-1).mean()).tolist()
    return R


def sam_scene(g, y):
    dot = (g * y).sum(0); n = np.linalg.norm(g, axis=0) * np.linalg.norm(y, axis=0); v = n > 0
    return float(np.degrees(np.arccos(np.clip(dot[v] / n[v], -1, 1))).mean())


def sam_shapley(target, gt, a, b, masks):
    group_masks = [np.logical_or.reduce([masks[x] for x in g]) for g in GROUPS]; m = len(GROUPS)
    w = [math.factorial(k) * math.factorial(m - k - 1) / math.factorial(m) for k in range(m)]; rows = []
    for s in SEEDS:
        ya, yb = cached(target, a, s), cached(target, b, s)
        for i in range(gt.shape[0]):
            D = np.fft.fft2(yb[i] - ya[i], axes=(-2, -1)); parts = [np.fft.ifft2(D * gm, axes=(-2, -1)).real for gm in group_masks]; val = {}
            for sub in itertools.product((0, 1), repeat=m):
                val[sub] = sam_scene(gt[i], ya[i] + sum((p for p, on in zip(parts, sub) if on), np.zeros_like(ya[i])))
            phi = np.zeros(m)
            for sub, v in val.items():
                for j in range(m):
                    if not sub[j]: phi[j] += w[sum(sub)] * (v - val[sub[:j] + (1,) + sub[j + 1:]])  # positive: adding j lowers SAM
            rows.append(phi)
    A = np.array(rows).reshape(len(SEEDS), gt.shape[0], m)
    return dict(groups=GROUP_NAMES, total_deg=float(A.sum(-1).mean()), by_group=A.mean((0, 1)).tolist(), seed=A.mean(1).tolist())


def run_frequency(log):
    for target in TARGETS:
        data = read_test(target); gt = data['gt']; masks = bin_masks(*gt.shape[-2:]); pairs = pairs_of(target)
        out = dict(target=target, bins=BINS, frequency=frequency_block(target, gt, data['lms'], arms_of(target), pairs, masks),
                   sam_shapley={f'{a}->{b}': sam_shapley(target, gt, a, b, masks) for a, b in pairs})
        write(f'frequency_{target}.json', out); log(f'frequency {target}')


def run_rr2(log):
    data = read_test('QB'); wald = P68.FWald('QB'); deg = [wald.degrade(data['pan'][i], data['ms'][i]) for i in range(len(data['pan']))]; gt = data['ms']
    means = {}
    for arm in arms_of('QB'):
        means[arm] = []
        for s in SEEDS:
            net = load_net('QB', arm, s); ys = np.stack([P68.predict(net, d, 2047.) for d in deg])
            np.save(CACHE / f'QBRR2_{arm}_s{s}.npy', ys.astype(np.float32))
            means[arm].append(float((10 * np.log10(2047 ** 2 / ((ys - gt) ** 2).mean((-1, -2)))).mean()))
        expected = CFG['reproduction']['qb_rr2_expected_mean_psnr'][arm]
        if abs(np.mean(means[arm]) - expected) > CFG['reproduction']['qb_rr2_tolerance_db']: raise ValueError(f'RR2 not reproduced {arm} {means[arm]}')
    out = dict(target='QBRR2', bins=BINS, mean_psnr=means,
               frequency=frequency_block('QBRR2', gt, np.stack([d[1] for d in deg]), arms_of('QB'), pairs_of('QB'), bin_masks(*gt.shape[-2:]), hann=False))
    write('frequency_QB_RR2.json', out); log(f'rr2 {means}')


def run_swap(log):
    """Cross check without the LMDI split: replace one side of a cutoff in the C0 output by C2MS and measure directly."""
    gt = read_test('QB')['gt']; r = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing='ij')); res = {}
    psnr = lambda y: float((10 * np.log10(2047 ** 2 / ((y - gt) ** 2).mean((-1, -2)))).mean())
    sam = lambda y: float(np.mean([sam_scene(gt[i], y[i]) for i in range(len(gt))]))
    for label, cut in (('16px', 1 / 16), ('8px', 1 / 8), ('4px', 1 / 4)):
        low, rows = r < cut, []
        for s in SEEDS:
            a, b = cached('QB', 'C0', s), cached('QB', 'C2MS', s); Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
            high_only = np.fft.ifft2(np.where(low, Fa, Fb)).real; low_only = np.fft.ifft2(np.where(low, Fb, Fa)).real
            rows.append([psnr(b) - psnr(a), psnr(high_only) - psnr(a), psnr(low_only) - psnr(a), sam(a) - sam(b), sam(a) - sam(high_only), sam(a) - sam(low_only)])
        res[label] = dict(zip(('psnr_all', 'psnr_high_only', 'psnr_low_only', 'sam_all', 'sam_high_only', 'sam_low_only'), np.mean(rows, 0).tolist()))
    write('band_swap_check.json', res); log(res)


# ---------------------------------------------------------------- structure

def scale_maps(I):
    M = []
    for s in SCALES:
        gy = ndi.gaussian_filter(I, s, order=(1, 0), mode='nearest'); gx = ndi.gaussian_filter(I, s, order=(0, 1), mode='nearest')
        M.append(s * np.hypot(gx, gy))
    return np.stack(M)


def coherence(I):
    c = S['coherence']
    gy = ndi.gaussian_filter(I, c['gradient_sigma'], order=(1, 0), mode='nearest'); gx = ndi.gaussian_filter(I, c['gradient_sigma'], order=(0, 1), mode='nearest')
    jxx, jyy, jxy = (ndi.gaussian_filter(a, c['integration_sigma'], mode='nearest') for a in (gx * gx, gy * gy, gx * gy))
    return np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(jxx + jyy, 1e-9)


def classify(I, gate, tau=TAU, M=None):
    """0 flat, 1 fine, 2 small, 3 medium, 4 large, 5 large and straight. The class is the last scale reached in an unbroken chain from
    s = 1 with M_s >= tau M_1; M_1 under the gate is flat."""
    M = scale_maps(I) if M is None else M
    cl = 1 + np.logical_and.accumulate(M[1:] >= tau * M[0], axis=0).sum(0)
    cl = np.where((cl == 4) & (coherence(I) >= COH), 5, cl)
    return np.where(M[0] < gate, 0, cl).astype(np.int8)


def run_check(log):
    H = 128; res = {}
    def blurred(img): return ndi.gaussian_filter(img.astype(float), 1.0, mode='nearest')
    cases = {}
    step = np.zeros((H, H)); step[:, H // 2:] = 100; cases['step edge'] = (blurred(step), np.abs(np.arange(H) - H // 2)[None, :].repeat(H, 0) <= 1)
    for w in (1, 2, 4, 8, 16):
        a = np.zeros((H, H)); a[:, H // 2 - w // 2:H // 2 - w // 2 + w] = 100
        m = np.zeros((H, H), bool); m[:, H // 2 - w // 2 - 1:H // 2 - w // 2 + w + 1] = True; cases[f'line width {w}'] = (blurred(a), m)
    for w in (2, 4, 8):
        a = np.zeros((H, H)); o = H // 2 - w // 2; a[o:o + w, o:o + w] = 100
        m = np.zeros((H, H), bool); m[o - 1:o + w + 1, o - 1:o + w + 1] = True; cases[f'square {w}x{w}'] = (blurred(a), m)
    for name, (img, mask) in cases.items():
        M = scale_maps(img); active = mask & (M[0] >= 0.3 * M[0].max()); cl = classify(img, 0.0, M=M)
        res[name] = {NAMES[j]: float((cl[active] == j).mean()) for j in range(1, 6)}
    write('structure_check.json', res); log(res)


def lmdi(sa, sb, mask):
    """sa, sb: (seed, scene, band, H, W) squared errors, mask (scene, H, W): dB contribution per (seed, scene)."""
    return K * (((sa - sb) * mask[:, None]).mean((-1, -2)) / log_mean(sa.mean((-1, -2)), sb.mean((-1, -2)))).mean(2)


def class_table(errs, cl, names, pairs):
    out = dict(names=names, pixel_share=[float((cl == k).mean()) for k in range(len(names))], arms={}, pairs={})
    sq = {a: e ** 2 for a, e in errs.items()}
    for arm, s in sq.items():
        se = s.mean(2); out['arms'][arm] = dict(mse=[float(se[:, cl == k].mean()) for k in range(len(names))], share_of_total_se=[float(se[:, cl == k].sum() / se.sum()) for k in range(len(names))])
    for a, b in pairs:
        C = np.stack([lmdi(sq[a], sq[b], cl == k) for k in range(len(names))], -1); lo, hi = boot(C.mean(0))  # (seed, scene, class)
        out['pairs'][f'{a}->{b}'] = dict(total_db=float(C.sum(-1).mean()), db=C.mean((0, 1)).tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(), share=(C.mean((0, 1)) / C.sum(-1).mean()).tolist(),
                                         db_seed=C.mean(1).tolist(), scene_db=C.mean(0).tolist(), scenes_improved=(C.mean(0) > 0).sum(0).tolist(),
                                         reduction_pct=[float(100 * (1 - sq[b].mean(2)[:, cl == k].mean() / sq[a].mean(2)[:, cl == k].mean())) for k in range(len(names))])
    return out


def error_stack(target, gt, arms):
    return {a: np.stack([(cached(target, a, s) - gt).astype(np.float32) for s in SEEDS]) for a in arms}


def run_structure(log):
    for target in TARGETS:
        gt = read_test(target)['gt']; arms, pairs = arms_of(target), pairs_of(target); n, _, h, w = gt.shape
        I = gt.mean(1); Ms = np.stack([scale_maps(i) for i in I]); gate = float(np.percentile(Ms[:, 0], GATE))
        cl = np.stack([classify(I[i], gate, M=Ms[i]) for i in range(n)]); errs = error_stack(target, gt, arms)
        R = dict(target=target, gate=gate, main=class_table(errs, cl, NAMES, pairs), sensitivity={})
        a, b = pairs[-1]
        for row in S['sensitivity']:
            g2 = float(np.percentile(Ms[:, 0], row['gate'])); c2 = np.stack([classify(I[i], g2, tau=row['tau'], M=Ms[i]) for i in range(n)])
            t = class_table({x: errs[x] for x in (a, b)}, c2, NAMES, ((a, b),))
            R['sensitivity'][f"tau{row['tau']}_gate{row['gate']}"] = dict(pixel_share=t['pixel_share'], pair=t['pairs'][f'{a}->{b}'])
        # equal contrast: pooled terciles of M_1 over the active pixels
        t = np.percentile(Ms[:, 0][cl > 0], [33.33, 66.67]); strat = np.digitize(Ms[:, 0], t); sa, sb = (errs[a] ** 2).mean(2), (errs[b] ** 2).mean(2); tab = {}
        for k in range(1, 5):
            mask_k = cl >= 4 if k == 4 else cl == k; label = 'large(s8)' if k == 4 else NAMES[k]
            for s in range(3):
                m = mask_k & (strat == s)
                tab[f'{label}_T{s + 1}'] = dict(pixel_share=float(m.mean()), mse_a=float(sa[:, m].mean()), mse_b=float(sb[:, m].mean()), reduction_pct=float(100 * (1 - sb[:, m].mean() / sa[:, m].mean())))
        R['size_by_contrast'] = dict(terciles=t.tolist(), table=tab)
        # class x band: pixel energy of each error band inside a class, and the detail gain y_band g_band / g_band^2 there
        r = np.hypot(*np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij'))
        parts = dict(zip(('>16px', '8-16px', '4-8px', '<4px'), (r < 1 / 16, (r >= 1 / 16) & (r < 1 / 8), (r >= 1 / 8) & (r < 1 / 4), r >= 1 / 4)))
        Fg = np.fft.fft2(gt.astype(np.float32), axes=(-2, -1)); X = {}
        for arm in arms:
            Fe = np.fft.fft2(errs[arm], axes=(-2, -1)); Fy = Fe + Fg; X[arm] = {}
            for p, m in parts.items():
                e = np.fft.ifft2(Fe * m, axes=(-2, -1)).real; y = np.fft.ifft2(Fy * m, axes=(-2, -1)).real; g = np.fft.ifft2(Fg * m, axes=(-2, -1)).real; row = {}
                for k, name in enumerate(NAMES):
                    mask = np.broadcast_to((cl == k)[:, None], g.shape)
                    row[name] = dict(mse=float((e ** 2).mean(0)[mask].mean()), gain=float((y * g[None]).mean(0)[mask].sum() / max((g ** 2)[mask].sum(), 1e-9)))
                X[arm][p] = row
            del Fe, Fy
        R['class_by_frequency'] = X
        # intensity and chroma: e = mean over bands + rest is an exact split of the band mean squared error
        idx = np.searchsorted(EDGES, r, side='right') - 1; idx[0, 0] = -1; L = {}
        for arm in arms:
            ebar = errs[arm].mean(2); lum = (ebar ** 2).mean((0, 1)); total = (errs[arm] ** 2).mean((0, 1, 2))
            Fi = np.abs(np.fft.fft2(ebar, axes=(-2, -1))) ** 2 / (h * w) ** 2; Ft = (np.abs(np.fft.fft2(errs[arm], axes=(-2, -1))) ** 2 / (h * w) ** 2).mean(2)
            L[arm] = dict(intensity_mse=float(lum.mean()), chroma_mse=float((total - lum).mean()), intensity_by_bin=[float(Fi[..., idx == j - 1].sum(-1).mean()) for j in range(len(BINS))],
                          total_by_bin=[float(Ft[..., idx == j - 1].sum(-1).mean()) for j in range(len(BINS))])
        R['intensity_chroma'] = L
        write(f'structure_{target}.json', R); log(f'structure {target}'); del errs


def thin_segments(edges):
    """Label 8-connected contour segments after removing crossings, so a street grid is not one long contour."""
    e = edges.copy(); e &= ~((ndi.convolve(e.astype(int), np.ones((3, 3), int), mode='constant') - e) >= 3)
    lab, n = ndi.label(e, structure=np.ones((3, 3))); size = np.bincount(lab.ravel(), minlength=n + 1); size[0] = 0
    return size[lab]


def run_canny(log):
    """The rejected first classes, kept so the rejection can be checked."""
    c = S['rejected_canny']; gt = read_test('QB')['gt']; I = gt.mean(1); n = len(I); names = ['flat', 'texture', 'short<16', 'medium16-47', 'long>=48']
    def grad(sigma, i): return np.hypot(ndi.gaussian_filter(I[i], sigma, order=(1, 0), mode='nearest'), ndi.gaussian_filter(I[i], sigma, order=(0, 1), mode='nearest'))
    def th(sigma): return tuple(np.percentile(np.stack([grad(sigma, i) for i in range(n)]), c['quantiles']))
    lo, hi = th(c['sigma']); near = lambda m, d=c['zone_radius']: ndi.distance_transform_edt(~m) <= d if m.any() else np.zeros_like(m)
    local = lambda x: np.sqrt(np.maximum(ndi.uniform_filter(x * x, 5, mode='nearest') - ndi.uniform_filter(x, 5, mode='nearest') ** 2, 0))
    edges = [canny(I[i], sigma=c['sigma'], low_threshold=lo, high_threshold=hi, mode='nearest') for i in range(n)]
    q = float(np.median(np.concatenate([local(I[i])[~near(edges[i])] for i in range(n)]))); A = []
    for i in range(n):
        length = thin_segments(edges[i]); cl = np.zeros(I[i].shape, np.int8); zone = near(edges[i])
        cl[~zone & (local(I[i]) >= q)] = 1; cl[zone] = 2; cl[near(length >= c['medium'])] = 3; cl[near(length >= c['long'])] = 4; A.append(cl)
    t1, t4 = th(1.0), th(4.0); B = []
    for i in range(n):
        e1 = canny(I[i], sigma=1.0, low_threshold=t1[0], high_threshold=t1[1], mode='nearest'); e4 = canny(I[i], sigma=4.0, low_threshold=t4[0], high_threshold=t4[1], mode='nearest')
        cl = np.zeros(I[i].shape, np.int8); keep = near(e4, 3); cl[near(e1 & ~keep)] = 1; cl[near(e1 & keep)] = 2; B.append(cl)
    errs = error_stack('QB', gt, ('C0', 'C2MS')); pairs = (('C0', 'C2MS'),)
    write('structure_rejected_canny.json', dict(contour_length=class_table(errs, np.stack(A), names, pairs), persistence=class_table(errs, np.stack(B), ['no_edge', 'fine_only', 'persistent_to_sigma4'], pairs)))
    log('canny')


def run_images(log):
    stats = result_images.render(load_net('QB', 'C0', 42), load_net('QB', 'C2MS', 42), ROOT / 'data', OUT / 'images/original_vs_C2MS', names=('원본', 'C2MS'), seed=42, device='cpu',
                                 labels=('원본 SSA-MRN (seed 42)', 'C2MS (seed 42)'))
    log(f'images {sorted(stats)}')


# ---------------------------------------------------------------- part: summary

def summary_numbers():
    res = {}
    for target in TARGETS + ('QB_RR2',):
        d = json.load(open(OUT / f'frequency_{target}.json', encoding='utf-8'))['frequency']; pair = 'C0->C2MS'; p = d['pairs'][pair]; b = np.array(d['C2MS']['mse_by_bin'])
        res[target] = dict(total_db=p['total_db'], total_db_seed=p['total_db_seed'], share_over_16px=sum(p['share'][:4]), share_8_16px=p['share'][4], share_under_8px=sum(p['share'][5:]),
                           residual_share_2_4px=float(b[6] / b.sum()), residual_share_4_8px=float(b[5] / b.sum()))
        if target != 'QB_RR2':
            m = json.load(open(OUT / f'structure_{target}.json', encoding='utf-8'))['main']; s = m['pairs'][pair]; px = np.array(m['pixel_share']); share = np.array(s['share'])
            res[target].update(pixel_share=px.tolist(), structure_share=share.tolist(), structure_density=(share / px).tolist(), structure_reduction_pct=s['reduction_pct'])
    return res


def run_summary(log):
    write('summary.json', summary_numbers())
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    for name in ('Apple SD Gothic Neo', 'AppleGothic', 'Malgun Gothic', 'NanumGothic'):
        if any(f.name == name for f in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = name; break
    plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    BLUE, GRAY, INK = '#2a78d6', '#b7b6ae', '#52514e'
    groups = ['16 px 초과', '8-16 px', '4-8 px', '2-4 px', '2 px 미만']; pick = lambda x: [sum(x[:4]), x[4], x[5], x[6], x[7]]
    Q = json.load(open(OUT / 'frequency_QB.json', encoding='utf-8'))['frequency']; M = json.load(open(OUT / 'structure_QB.json', encoding='utf-8'))['main']

    # gain by band and structure on QB, with the band shares of the other tests
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    y = np.arange(5); share = 100 * np.array(pick(Q['pairs']['C0->C2MS']['share']))
    ax[0].barh(y, share, color=BLUE, height=.62); ax[0].set_yticks(y, groups); ax[0].invert_yaxis(); ax[0].set_xlabel('PSNR 이득의 몫 (%)')
    ax[0].set_title(f"주파수 대역 (QB, +{Q['pairs']['C0->C2MS']['total_db']:.3f} dB)", loc='left', fontsize=11)
    for i, v in enumerate(share): ax[0].text(v + .8, i, f'{v:.1f}', va='center', color=INK)
    labels = ['미세 구조', '작은 세부', '중간 구조', '큰 구조·외곽선', '평탄 영역']; s = np.array(M['pairs']['C0->C2MS']['share']); px = np.array(M['pixel_share'])
    gain = 100 * np.array([s[1], s[2], s[3], s[4] + s[5], s[0]]); area = 100 * np.array([px[1], px[2], px[3], px[4] + px[5], px[0]]); y = np.arange(5)
    ax[1].barh(y - .19, gain, color=BLUE, height=.34, label='이득의 몫'); ax[1].barh(y + .19, area, color=GRAY, height=.34, label='화소 비율')
    for i, (g_, a_) in enumerate(zip(gain, area)): ax[1].text(g_ + .6, i - .19, f'{g_:.1f}', va='center', color=INK); ax[1].text(a_ + .6, i + .19, f'{a_:.1f}', va='center', color=INK)
    ax[1].set_yticks(y, labels); ax[1].invert_yaxis(); ax[1].set_xlim(0, 56); ax[1].set_xlabel('비율 (%)'); ax[1].legend(frameon=False, loc='upper right'); ax[1].set_title('구조 크기 (QB)', loc='left', fontsize=11)
    names = {'QB': 'QB', 'GF2': 'GF2', 'WV3': 'WV3', 'QB_RR2': 'QB RR2', 'WV2': 'WV2 전이'}; S_ = summary_numbers(); y = np.arange(len(names)); left = np.zeros(len(names))
    for key, label, color in (('share_under_8px', '8 px 미만', BLUE), ('share_8_16px', '8-16 px', '#8fb8ea'), ('share_over_16px', '16 px 초과', GRAY)):
        v = 100 * np.array([S_[k][key] for k in names]); ax[2].barh(y, v, left=left, color=color, height=.62, label=label, edgecolor='white', linewidth=2); left += v
    ax[2].set_yticks(y, list(names.values())); ax[2].invert_yaxis(); ax[2].set_xlabel('PSNR 이득의 몫 (%)'); ax[2].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.2)); ax[2].set_title('데이터별 주파수 구성', loc='left', fontsize=11)
    fig.savefig(OUT / 'gain_breakdown.png', dpi=150, facecolor='white'); plt.close(fig)

    # what the bands and the classes look like on the fixed scenes
    data = read_test('QB'); gt = data['gt']; I = gt.mean(1); Ms = np.stack([scale_maps(i) for i in I]); gate = float(np.percentile(Ms[:, 0], GATE))
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing='ij')); scene = 3; F = np.fft.fft2(I[scene])
    bands = (('원본 정답 밝기', None), ('16 px 초과 (큰 형태)', r < 1 / 16), ('8-16 px', (r >= 1 / 16) & (r < 1 / 8)), ('4-8 px', (r >= 1 / 8) & (r < 1 / 4)), ('4 px 미만', r >= 1 / 4))
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.1), constrained_layout=True)
    for a, (name, mask) in zip(ax, bands):
        img = I[scene] if mask is None else np.fft.ifft2(F * mask).real
        if mask is None or name.startswith('16'): lo, hi = np.percentile(I[scene], (1, 99))
        else: hi = np.percentile(np.abs(img), 99.5); lo = -hi
        a.imshow(img, cmap='gray', vmin=lo, vmax=hi); a.set_title(name, fontsize=10); a.axis('off')
    fig.savefig(OUT / 'frequency_bands_scene03.png', dpi=130, facecolor='white'); plt.close(fig)
    scenes = (3, 7, 14); colors = ListedColormap(['#ecebe6', '#c8102e', '#f08a24', '#2a78d6', '#1baf7a', '#0b5a36'])
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
    bands_rgb = list(result_images.BANDS)
    for j, i in enumerate(scenes):
        rgb = gt[i][bands_rgb].transpose(1, 2, 0); lo, hi = np.percentile(rgb, (1, 99)); ax[0, j].imshow(np.clip((rgb - lo) / (hi - lo), 0, 1)); ax[0, j].set_title(f'장면 {i} 정답', fontsize=10); ax[0, j].axis('off')
        ax[1, j].imshow(classify(I[i], gate, M=Ms[i]), cmap=colors, vmin=-.5, vmax=5.5, interpolation='nearest'); ax[1, j].axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors(k)) for k in range(6)]
    fig.legend(handles, ['평탄', '미세', '작은 세부', '중간', '큰 구조', '큰 직선 외곽선'], loc='outside lower center', ncol=6, frameon=False)
    fig.savefig(OUT / 'structure_classes.png', dpi=130, facecolor='white'); plt.close(fig)
    log('summary')


PARTS = dict(check=run_check, predict=run_predict, frequency=run_frequency, rr2=run_rr2, swap=run_swap, structure=run_structure, canny=run_canny, images=run_images, summary=run_summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--part', choices=tuple(PARTS), nargs='*', default=tuple(PARTS))
    parser.add_argument('--cache', default=str(Path(tempfile.gettempdir()) / 'ctrs72_predictions'), help='folder for the prediction arrays, never archived')
    args = parser.parse_args(); CACHE = Path(args.cache); CACHE.mkdir(parents=True, exist_ok=True)
    for part in args.part: PARTS[part](lambda m: print(m, flush=True))
