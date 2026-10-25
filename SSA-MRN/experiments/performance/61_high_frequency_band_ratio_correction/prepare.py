"""Experiment 61 phase 0 without training: is the C2MS high-band angle error in its band ratio or in its strength?

Uses the experiment 60 loaders (SHA-256 checked data, archived C2MS best.pt, CPU FP32 inference, sensor MTF lowpass L) and
the same 20 RR scenes per target. With H = y - L(y), O1 gives the high band the GT band ratio at the C2MS strength, O2 the
GT strength at the C2MS band ratio, O3 is a rule without GT that keeps only the part of H along the local low-band color.
The Ga sensitivity rows re-run the experiment 60 cross-fitted oracle with larger windows and a diagonal map (reported only).
Gates are fixed in config.json before measurement. Writes only below results/.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location('exp60_prepare', HERE.parent / '60_norm_preserving_spectral_direction_refinement/prepare.py')
P60 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P60)
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def unit(x):
    n = np.linalg.norm(x, axis=0, keepdims=True)
    return np.where(n > 0, x / np.where(n > 0, n, 1), 0), n


def windowed_oracle(y, g, window, diagonal=False):
    """Cross-fitted (checkerboard halves) band map y -> g per window; affine B x (B+1), or per-band gain + bias; keeps |y|."""
    c, h, w = y.shape; out = np.empty_like(y)
    rows, cols = np.indices((window, window)); board = ((rows + cols) % 2 == 0).ravel()
    for i in range(0, h, window):
        for j in range(0, w, window):
            yw = y[:, i:i + window, j:j + window].reshape(c, -1).T; gw = g[:, i:i + window, j:j + window].reshape(c, -1).T
            fit = np.empty_like(gw)
            for half in (board, ~board):
                if diagonal:
                    for b in range(c):
                        x = np.stack([yw[half, b], np.ones(half.sum())], axis=1)
                        coef = np.linalg.lstsq(x, gw[half, b], rcond=None)[0]
                        fit[~half, b] = yw[~half, b] * coef[0] + coef[1]
                else:
                    x = np.concatenate([yw, np.ones((len(yw), 1))], axis=1)
                    fit[~half] = x[~half] @ np.linalg.lstsq(x[half], gw[half], rcond=None)[0]
            out[:, i:i + window, j:j + window] = fit.T.reshape(c, window, window)
    return P60.with_norm(out, np.linalg.norm(y, axis=0, keepdims=True))


def scene(y, a, kernel):
    g = a['gt']; ly, lg = P60.lowpass(y, kernel), P60.lowpass(g, kernel)
    hy, hg = y - ly, g - lg
    ry, ny = unit(hy); rg, ng = unit(hg); u, _ = unit(ly)
    variants = {
        'C2MS': y,
        'O1_hf_ratio': ly + np.where(ng > 0, ny * rg, hy),
        'O2_hf_strength': ly + np.where(ny > 0, ng * ry, hy),
        'O3_hf_along_lf_rule': ly + np.sum(hy * u, axis=0, keepdims=True) * u,
        'HF_fixed': ly + hg,
        'Ga_affine_16': windowed_oracle(y, g, 16),
        'Ga_affine_32': windowed_oracle(y, g, 32),
        'Ga_diagonal_8': windowed_oracle(y, g, 8, diagonal=True),
    }
    # how much of the high band is along the low-band color, for C2MS and for the GT
    share = lambda h, n: float(np.sum(np.sum(h * u, axis=0) ** 2) / max(np.sum(n ** 2), 1e-12))
    return dict(metrics={k: {m: float(v[m]) for m in ('PSNR', 'SAM', 'ERGAS')} for k, v in ((k, rr_metrics(g, v)) for k, v in variants.items())},
                hf_energy_along_lf={'C2MS': share(hy, ny), 'GT': share(hg, ng)},
                hf_strength_ratio=float(np.sum(ny) / max(np.sum(ng), 1e-12)),
                hf_ratio_angle_deg=float(np.nanmean(P60.angles(hy, hg))))


def plot(result, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    targets = list(result['targets']); variants = ('C2MS', 'O1_hf_ratio', 'O2_hf_strength', 'O3_hf_along_lf_rule', 'Ga_affine_32', 'Ga_diagonal_8')
    fig, ax = plt.subplots(figsize=(11, 4.2)); width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [result['targets'][t]['seed_mean']['metrics'][v]['SAM'] for t in targets]
        ax.bar(np.arange(len(targets)) + (i - (len(variants) - 1) / 2) * width, vals, width, label=v)
    ax.set_xticks(range(len(targets)), targets); ax.set_ylabel('SAM (deg), seed mean'); ax.legend(fontsize=8)
    ax.set_title('C2MS high band: GT band ratio (O1) vs GT strength (O2), rule O3, Ga sensitivity')
    fig.tight_layout(); fig.savefig(out / 'phase0.png', dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--targets', nargs='+', default=list(CFG['targets']))
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--plot-only', action='store_true')
    a = p.parse_args(); torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    if a.plot_only:
        plot(json.loads((a.out / 'phase0.json').read_text(encoding='utf-8')), a.out); return
    a.out.mkdir(parents=True, exist_ok=True); started = time.time(); result = {'targets': {}, 'models': {}}
    for target in a.targets:
        sensor = CFG['targets'][target]; scenes = P60.load(a.data, target); scale = P60.SENSORS[target]['max_value']
        kernel = torch.from_numpy(np.stack([P60.gen_mtf(gn, 4) for gn in P60.SENSORS[target]['ms_gnyq']])[:, None]).double()
        per_seed = {}
        for seed in CFG['seeds']:
            net, path, digest = P60.net_for(sensor, seed)
            result['models'][f'C2MS_{sensor}_s{seed}'] = dict(checkpoint=path, sha256=digest)
            rows = [scene(P60.predict(net, s, scale), s, kernel) for s in scenes]
            m = P60.mean_dicts(rows); ref = P60.reference(target, seed)
            m['reproduction'] = {k: m['metrics']['C2MS'][k] - ref[k] for k in ref}
            if any(abs(v) > 1e-3 for v in m['reproduction'].values()):
                raise ValueError(f'{target} s{seed} does not reproduce the archived RR means: {m["reproduction"]}')
            m['scenes'] = [r['metrics'] for r in rows]; per_seed[seed] = m
            print(target, seed, {k: round(v['SAM'], 4) for k, v in m['metrics'].items()}, flush=True)
        result['targets'][target] = dict(seed_mean=P60.mean_dicts([{k: v for k, v in per_seed[s].items() if k != 'scenes'} for s in CFG['seeds']]),
                                         seeds={str(s): per_seed[s] for s in CFG['seeds']})
    if 'QB' in result['targets']:
        mq = result['targets']['QB']['seed_mean']['metrics']; c = mq['C2MS']
        dec = lambda k: c['SAM'] - mq[k]['SAM']
        g3_psnr = mq['O3_hf_along_lf_rule']['PSNR'] - c['PSNR']
        result['gates'] = {'G1_ratio': dict(sam_decrease=dec('O1_hf_ratio'), passed=bool(dec('O1_hf_ratio') >= 0.50)),
                           'G2_strength': dict(sam_decrease=dec('O2_hf_strength'), passed=bool(dec('O2_hf_strength') >= 0.50)),
                           'G3_rule': dict(sam_decrease=dec('O3_hf_along_lf_rule'), psnr_change=g3_psnr,
                                           passed=bool(dec('O3_hf_along_lf_rule') >= 0.10 and g3_psnr >= -0.04))}
        g = result['gates']
        g['phase1'] = 'band-ratio head' if g['G1_ratio']['passed'] else 'strength head' if g['G2_strength']['passed'] else None
    result['environment'] = dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=torch.get_num_threads(),
                                 seconds=round(time.time() - started, 1), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (a.out / 'phase0.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(result, a.out)
    print(json.dumps(result.get('gates'), indent=1))


if __name__ == '__main__':
    main()
