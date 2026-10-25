"""Experiment 60 phase 0 without training: where the C2MS spectral angle error lives, and two gates.

Runs the archived C2MS best.pt (53 QB, 54 GF2/WV3; WV3 models also on WV2) on all 20 RR test scenes and measures, per pixel,
the angle between the prediction and the GT band vectors (the quantity that common/metrics.rr_metrics averages as SAM).
D1-D6 split that angle by GT norm, PAN gradient, NDVI and frequency band (config.json phase0.diagnostics).
Ga (cross-fitted local 8x8 affine direction oracle) and Gb (C2MS norm x GT direction) are the gates fixed in config.json
before measurement. Both oracles use the GT and are ceilings, not models. Writes only below results/.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics  # noqa: E402

E54 = HERE.parent / '54_multi_sensor_scale_mixed_training'


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


Model = _module('exp54_model', E54 / 'model.py').Model  # 54 Model: C2MS for any band count, same core as 53
P54 = _module('exp54_prepare', E54 / 'prepare.py')
SENSORS, _blur, gen_mtf = P54.SENSORS, P54._blur, P54.gen_mtf

CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
CFG54 = json.loads((E54 / 'config.json').read_text(encoding='utf-8'))
REF = {'QB': json.loads((HERE.parent / '53_scale_mixed_training/results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models'],
       'OTHER': json.loads((E54 / 'results/evaluation_v2_3.json').read_text(encoding='utf-8'))['models']}
RED_NIR = {'QB': (2, 3), 'GF2': (2, 3), 'WV3': (4, 6), 'WV2': (4, 6)}
WINDOW = 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def expected_sha(target):
    if target == 'QB': return CFG['data_sha256']['QB']
    table = CFG54['data_sha256'] if target in CFG54['data_sha256'] else CFG54['transfer_data_sha256']
    return table[target]['RR']


def load(data, target):
    path = data / 'test' / f"test_{SENSORS[target]['tag']}_multiExm1.h5"
    if sha(path) != expected_sha(target): raise ValueError(f'data sha mismatch {target}')
    with h5py.File(path, 'r') as f:
        return [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(len(f['pan']))]


def net_for(sensor, seed):
    path = ROOT / CFG['checkpoints'][sensor].replace('<seed>', str(seed))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = Model('C2MS', SENSORS[sensor]['bands'], seed)
    net.load_state_dict(ck['model'], strict=True)
    return net.eval(), str(path.relative_to(ROOT)), sha(path)


def predict(net, a, scale):
    x = [torch.from_numpy((a[k] / scale).astype('float32')).unsqueeze(0) for k in ('pan', 'lms', 'ms')]
    with torch.no_grad(): y = net(*x)
    out = y[0].float().numpy().astype(np.float64) * scale
    if not np.isfinite(out).all(): raise FloatingPointError('nonfinite prediction')
    return out


def angles(y, g):
    """Per-pixel angle in degrees (same formula and validity mask as common/metrics.rr_metrics)."""
    dot = np.sum(y * g, axis=0); norms = np.linalg.norm(y, axis=0) * np.linalg.norm(g, axis=0)
    valid = norms > 0
    ang = np.full(dot.shape, np.nan)
    ang[valid] = np.degrees(np.arccos(np.clip(dot[valid] / norms[valid], -1, 1)))
    return ang


def lowpass(x, kernel):
    return _blur(torch.from_numpy(x)[None], kernel)[0].numpy()


def with_norm(direction, norm):
    n = np.linalg.norm(direction, axis=0, keepdims=True)
    return np.where(n > 0, direction / np.where(n > 0, n, 1), 0) * norm


def local_direction_oracle(y, g):
    """Ga: cross-fitted 8x8 affine band map y -> g; fit on one checkerboard half, apply to the other; keep |y|."""
    c, h, w = y.shape; out = np.empty_like(y)
    rows, cols = np.indices((WINDOW, WINDOW)); board = (rows + cols) % 2 == 0
    for i in range(0, h, WINDOW):
        for j in range(0, w, WINDOW):
            yw = y[:, i:i + WINDOW, j:j + WINDOW].reshape(c, -1).T; gw = g[:, i:i + WINDOW, j:j + WINDOW].reshape(c, -1).T
            x = np.concatenate([yw, np.ones((len(yw), 1))], axis=1); fit = np.empty_like(gw)
            for half in (board.ravel(), ~board.ravel()):
                coef = np.linalg.lstsq(x[half], gw[half], rcond=None)[0]  # minimum-norm solution if a half is rank deficient
                fit[~half] = x[~half] @ coef
            out[:, i:i + WINDOW, j:j + WINDOW] = fit.T.reshape(c, WINDOW, WINDOW)
    return with_norm(out, np.linalg.norm(y, axis=0, keepdims=True))


def sobel(pan):
    p = np.pad(pan[0], 1, mode='edge')
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def split(ang, key, edges):
    """Mean angle, pixel share and share of the angle sum for bins of key (edges inclusive on the left)."""
    valid = np.isfinite(ang); total = ang[valid].sum(); n = valid.sum(); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (key >= lo) & (key < hi)
        out.append(dict(mean_angle=float(ang[m].mean()) if m.any() else None, pixel_share=float(m.sum() / n),
                        sam_share=float(ang[m].sum() / total)))
    return out


def scene(y, a, sensor, kernel):
    g, pan = a['gt'], a['pan']; ang = angles(y, g)
    norm_g = np.linalg.norm(g, axis=0); norm_y = np.linalg.norm(y, axis=0)
    red, nir = RED_NIR[sensor]; ndvi = (g[nir] - g[red]) / np.maximum(g[nir] + g[red], 1e-12)
    grad = sobel(pan); valid = np.isfinite(ang)
    variants = {
        'C2MS': y,
        'LF_fixed': y - lowpass(y, kernel) + lowpass(g, kernel),
        'HF_fixed': lowpass(y, kernel) + g - lowpass(g, kernel),
        'LMS': a['lms'],
        'Ga_local_direction_oracle': local_direction_oracle(y, g),
        'Gb_direction_ceiling': with_norm(g, norm_y[None]),
    }
    metrics = {k: rr_metrics(g, v) for k, v in variants.items()}
    weight = norm_y * norm_g
    return dict(
        metrics={k: {m: float(v[m]) for m in ('PSNR', 'SAM', 'ERGAS')} for k, v in metrics.items()},
        D1_norm_deciles=split(ang, norm_g, np.quantile(norm_g[valid], np.linspace(0, 1, 11)) + np.r_[np.zeros(10), 1e-9]),
        D2_pan_gradient=split(ang, grad, np.r_[np.quantile(grad, [0, 1 / 3, 2 / 3]), grad.max() + 1e-9]),
        D3_ndvi=split(ang, ndvi, [-np.inf, 0., 0.3, np.inf]),
        D6_corr_weight_angle=float(np.corrcoef(weight[valid], ang[valid])[0, 1]),
        D6_spearman_weight_angle=float(np.corrcoef(np.argsort(np.argsort(weight[valid])), np.argsort(np.argsort(ang[valid])))[0, 1]))


def mean_dicts(rows):
    if isinstance(rows[0], dict): return {k: mean_dicts([r[k] for r in rows]) for k in rows[0]}
    if isinstance(rows[0], list): return [mean_dicts([r[i] for r in rows]) for i in range(len(rows[0]))]
    vals = [r for r in rows if r is not None]
    return float(np.mean(vals)) if vals else None


def reference(target, seed):
    key = f'C2MS_s{seed}' if target == 'QB' else f"C2MS_{CFG['targets'][target]}_s{seed}"
    m = REF['QB'][key] if target == 'QB' else REF['OTHER'][key]['targets'][target]
    m = m.get('mean', m)
    return {k: m['RR'][k] for k in ('PSNR', 'SAM')}


def plot(result, out):
    """phase0.png: SAM of each variant per target (seed mean) and the mean angle per GT norm decile."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    targets = list(result['targets']); variants = ('C2MS', 'LF_fixed', 'Ga_local_direction_oracle', 'HF_fixed', 'LMS')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [result['targets'][t]['seed_mean']['metrics'][v]['SAM'] for t in targets]
        ax[0].bar(np.arange(len(targets)) + (i - (len(variants) - 1) / 2) * width, vals, width, label=v)
    ax[0].set_xticks(range(len(targets)), targets); ax[0].set_ylabel('SAM (deg), seed mean'); ax[0].legend(fontsize=8); ax[0].set_title('SAM after replacing parts of the C2MS output')
    for t in targets:
        ax[1].plot(range(1, 11), [r['mean_angle'] for r in result['targets'][t]['seed_mean']['D1_norm_deciles']], marker='o', label=t)
    ax[1].set_xlabel('GT norm decile (1 = darkest)'); ax[1].set_ylabel('mean pixel angle (deg)'); ax[1].legend(); ax[1].set_title('C2MS angle by GT norm')
    fig.tight_layout(); fig.savefig(out / 'phase0.png', dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--targets', nargs='+', default=list(CFG['targets']))
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--plot-only', action='store_true', help='redraw phase0.png from an existing phase0.json')
    a = p.parse_args(); torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    if a.plot_only:
        plot(json.loads((a.out / 'phase0.json').read_text(encoding='utf-8')), a.out); return
    a.out.mkdir(parents=True, exist_ok=True); started = time.time(); result = {'targets': {}, 'models': {}}
    for target in a.targets:
        sensor = CFG['targets'][target]; scenes = load(a.data, target); scale = SENSORS[target]['max_value']
        kernel = torch.from_numpy(np.stack([gen_mtf(gn, 4) for gn in SENSORS[target]['ms_gnyq']])[:, None]).double()
        per_seed = {}
        for seed in CFG['seeds']:
            net, path, digest = net_for(sensor, seed)
            result['models'][f'C2MS_{sensor}_s{seed}'] = dict(checkpoint=path, sha256=digest)
            rows = [scene(predict(net, s, scale), s, target, kernel) for s in scenes]
            m = mean_dicts(rows); ref = reference(target, seed)
            m['reproduction'] = {k: m['metrics']['C2MS'][k] - ref[k] for k in ref}
            if any(abs(v) > CFG['reference_rr_means']['tolerance'][k] for k, v in m['reproduction'].items()):
                raise ValueError(f'{target} s{seed} does not reproduce the archived RR means: {m["reproduction"]}')
            m['scenes'] = [r['metrics'] for r in rows]; per_seed[seed] = m
            print(target, seed, {k: round(v['SAM'], 4) for k, v in m['metrics'].items()}, flush=True)
        seeds_mean = mean_dicts([{k: v for k, v in per_seed[s].items() if k != 'scenes'} for s in CFG['seeds']])
        result['targets'][target] = dict(seed_mean=seeds_mean, seeds={str(s): per_seed[s] for s in CFG['seeds']})
    if 'QB' in result['targets']:
        mq = result['targets']['QB']['seed_mean']['metrics']; gates = CFG['phase0']['gates']
        ga = mq['C2MS']['SAM'] - mq['Ga_local_direction_oracle']['SAM']; gb = mq['Gb_direction_ceiling']['PSNR'] - mq['C2MS']['PSNR']
        result['gates'] = {'Ga': dict(sam_decrease=ga, threshold=0.50, passed=bool(ga >= 0.50)),
                           'Gb': dict(psnr_gain=gb, threshold=0.30, passed=bool(gb >= 0.30))}
        result['gates']['phase1'] = result['gates']['Ga']['passed'] and result['gates']['Gb']['passed']
    result['environment'] = dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=torch.get_num_threads(),
                                 seconds=round(time.time() - started, 1), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (a.out / 'phase0.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(result, a.out)
    print(json.dumps(result.get('gates'), indent=1))


if __name__ == '__main__':
    main()
