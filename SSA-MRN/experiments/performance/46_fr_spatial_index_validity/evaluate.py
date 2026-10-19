"""Experiment 46 additional evaluation (no training): original-QNR-paper variant of D_lambda / D_s.

Same 37 candidates per RR test scene as prepare.py. Indices of the DLPan flag_orig_paper=1 branch:
D_lambda_orig compares fused band-pair UQI (32x32 blocks) with LR MS band-pair UQI (8x8 blocks);
D_s_orig compares UQI(F_b, PAN) (32x32) with UQI(MS_LR_b, imresize(PAN, 1/4)) (8x8).
The toolbox 1.0 indices are recomputed to confirm the candidates match results/validity.json.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from scipy.stats import spearmanr
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp46_prepare', HERE / 'prepare.py')
P46 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P46)
from common.metrics import _block_uqi


def cubic(x):
    ax = np.abs(x); ax2, ax3 = ax ** 2, ax ** 3
    return (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))


def imresize_weights(n_in, scale):
    """MATLAB imresize contributions(): bicubic, antialiasing for scale < 1, symmetric boundary."""
    n_out = int(np.ceil(n_in * scale)); width = 4.0 / scale
    x = np.arange(1, n_out + 1); u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - width / 2); taps = int(np.ceil(width)) + 2
    ind = left[:, None] + np.arange(taps)[None, :] - 1
    w = scale * cubic(scale * (u[:, None] - ind - 1)); w /= w.sum(1, keepdims=True)
    aux = np.r_[np.arange(n_in), np.arange(n_in - 1, -1, -1)]
    ind = aux[np.mod(ind.astype(int), 2 * n_in)]
    W = np.zeros((n_out, n_in))
    for i in range(n_out): np.add.at(W[i], ind[i], w[i])
    return W


def imresize_matlab(img, scale):
    return imresize_weights(img.shape[0], scale) @ img @ imresize_weights(img.shape[1], scale).T


def orig_indices(fused, ms, pan, pan_lr):
    c = fused.shape[0]
    dl = np.mean([abs(_block_uqi(fused[i], fused[j], 32) - _block_uqi(ms[i], ms[j], 8)) for i in range(c) for j in range(i + 1, c)])
    ds = np.mean([abs(_block_uqi(fused[b], pan, 32) - _block_uqi(ms[b], pan_lr, 8)) for b in range(c)])
    return float(dl), float(ds), float((1 - dl) * (1 - ds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='results/validity.json of the prepare.py run')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    reference = json.loads(a.reference.read_text())
    found = list(a.data.rglob('test_qb_multiExm1.h5')); assert len(found) == 1
    n = a.limit or 20
    proj = P46.P45.Projector(a.device)
    arms, seeds = ('C0', 'C1', 'C2', 'C3'), (42, 43, 44)
    if a.limit: arms, seeds = ('C0', 'C2'), (42,)
    nets = {}
    for arm in arms:
        for seed in seeds:
            ck = P46.P45.E43 / f'checkpoints/{arm}_QB_s{seed}/best.pt'
            net = P46.P45.M43.Model(arm, seed); net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'], strict=True)
            nets[f'{arm}_s{seed}'] = net.to(a.device).eval()
    scenes, max_diff = [], 0.0
    with h5py.File(found[0], 'r') as f:
        for i in range(n):
            arr = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
            x = [torch.from_numpy((arr[k] / 2047).astype('float32')).unsqueeze(0).to(a.device) for k in ('pan', 'lms', 'ms')]
            out = {'GT': arr['gt'], 'LMS': arr['lms'], 'MTF_GLP': P46.P36.mtf_glp(arr['lms'], arr['pan'][0])}
            for key, net in nets.items():
                with torch.no_grad():
                    y = net(*x).float()
                yp, _ = proj.project(y, x[2], 5)
                out[key] = y[0].cpu().numpy().astype(np.float64) * 2047
                out[key + '_P5'] = yp[0].detach().cpu().numpy().astype(np.float64) * 2047
            for key in ('C0_s42', 'C2_s42'):
                if key not in out: continue
                for t in (0.25, 0.5, 0.75): out[f'{key}_toGT{t}'] = (1 - t) * out[key] + t * arr['gt']
                for t in (0.25, 0.5): out[f'{key}_toLMS{t}'] = (1 - t) * out[key] + t * arr['lms']
            pan = arr['pan'][0]
            pan_lr = {'matlab': imresize_matlab(pan, 0.25), 'skimage': resize(pan, arr['ms'].shape[1:], order=3, mode='reflect', anti_aliasing=True, preserve_range=True)}
            rows = {}
            for k, v in out.items():
                base = P46.indices(v, arr['gt'], arr['lms'], arr['ms'], arr['pan'])
                ref = reference['scenes'][i]['candidates'][k]
                max_diff = max(max_diff, max(abs(base[m] - ref[m]) for m in ('MSE', 'D_s', 'D_lambda')))
                row = {'MSE': base['MSE'], 'PSNR': base['PSNR'], 'D_s_toolbox': base['D_s'], 'D_lambda_toolbox': base['D_lambda'], 'QNR_toolbox': base['QNR']}
                for tag, plr in pan_lr.items():
                    dl, ds, q = orig_indices(v, arr['ms'], pan, plr)
                    row[f'D_lambda_orig_{tag}'], row[f'D_s_orig_{tag}'], row[f'QNR_orig_{tag}'] = dl, ds, q
                rows[k] = row
            scenes.append({'index': i, 'candidates': rows})
            print(i, 'GT Ds_orig %.4f C0 Ds_orig %.4f LMS Ds_orig %.4f | candidates match prepare.py (max diff %.2e)' % (rows['GT']['D_s_orig_matlab'], rows.get('C0_s42', rows['GT'])['D_s_orig_matlab'], rows['LMS']['D_s_orig_matlab'], max_diff), flush=True)
    names = list(scenes[0]['candidates'])
    c0 = [k for k in names if k.startswith('C0_s') and k.count('_') == 1]
    better = lambda idx: 1 if idx.startswith('QNR') else -1
    tests = {}
    for idx in [k for k in scenes[0]['candidates']['GT'] if k.startswith(('D_s', 'D_lambda', 'QNR'))]:
        sign = better(idx)
        t1 = sum(sign * (s['candidates']['GT'][idx] - np.mean([s['candidates'][k][idx] for k in c0])) > 0 for s in scenes)
        rhos = [spearmanr([s['candidates'][k]['MSE'] for k in names], [s['candidates'][k][idx] for k in names])[0] * (-sign) for s in scenes]
        t3 = {key: int(sum(sign * (s['candidates'][f'{key}_toGT0.75'][idx] - s['candidates'][key][idx]) > 0 for s in scenes)) for key in ('C0_s42', 'C2_s42') if f'{key}_toGT0.75' in names}
        tests[idx] = {'T1_GT_better_than_C0_mean': int(t1), 'T2_median_rank_corr_error_vs_badness': float(np.median(rhos)), 'T3_blend_to_GT_better': t3,
                      'GT_best_scenes': int(sum((min if sign < 0 else max)(s['candidates'], key=lambda k: s['candidates'][k][idx]) == 'GT' for s in scenes))}
    m = len(scenes); t = tests['D_s_orig_matlab']
    decision = ('conflicts' if t['T1_GT_better_than_C0_mean'] <= m // 4 and t['T2_median_rank_corr_error_vs_badness'] <= -0.5 else
                'consistent' if t['T1_GT_better_than_C0_mean'] >= m * 3 // 4 and t['T2_median_rank_corr_error_vs_badness'] >= 0.5 else 'mixed')
    means = {k: {idx: float(np.mean([s['candidates'][k][idx] for s in scenes])) for idx in scenes[0]['candidates'][k]} for k in names}
    env = {'python': sys.version, 'torch': torch.__version__, 'source_sha256': P46.sha(__file__), 'prepare_sha256': P46.sha(HERE / 'prepare.py'), 'config_sha256': P46.sha(HERE / 'config.json'),
           'reference_sha256': P46.sha(a.reference), 'candidates_max_abs_diff_vs_reference': max_diff}
    P46.write(a.output / 'orig_paper.json', {'environment': env, 'tests': tests, 'decision_D_s_orig': decision, 'means': means, 'scenes': scenes})
    P46.write(a.output / 'orig_complete.json', {'decision_D_s_orig': decision, 'training_started': False, 'time': time.time(), 'limit': a.limit})
    print('decision', decision, json.dumps(tests), flush=True)


if __name__ == '__main__':
    main()
