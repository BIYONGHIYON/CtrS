"""Experiment 36 phase 0 without training.

Ga: classical MTF-GLP (DLPan-Toolbox MTF_GLP.m reimplemented) vs archived A0 on RR20/FR20.
Gb: oracle ceiling of LMS + per-cell gains x fixed PAN detail bands on validation, with a null control.
Thresholds are fixed in config.json before measurement.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import h5py, numpy as np
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics, fr_metrics, _interp23tap

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
GNYQ_MS = [0.34, 0.32, 0.30, 0.22]
GNYQ_PAN = 0.15
A0 = ROOT / 'experiments/ssai/4_aligned_sigmoid_gating/results/evaluation/A0'
A0_VALIDATION = {'E_A': 1.7155e-4, 'source': 'experiment 35 phase 0 log, A0 best on the same 1,905 validation patches'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def kernel(gain, ratio, n=41, toolbox_lpfilter=False):
    """genMTF/MTF_PAN (alpha from (N-1)*fcut/2) or LPfilterGauss (alpha from N*fcut/2); Kaiser fwind1."""
    fcut = 1 / ratio
    span = n if toolbox_lpfilter else n - 1
    alpha = np.sqrt((span * fcut / 2) ** 2 / (-2 * np.log(gain)))
    c = np.arange(n) - (n - 1) / 2
    y, x = np.meshgrid(c, c, indexing='ij')
    window = np.interp(np.sqrt(x * x + y * y), np.arange(n // 2 + 1), np.kaiser(n, 0.5)[n // 2:], right=0.0)
    desired = np.exp(-(x * x + y * y) / (2 * alpha * alpha))
    desired[desired < np.finfo(float).eps * desired.max()] = 0
    desired /= desired.max()
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * window)


def imfilter(img, h):
    return correlate(img, h, mode='nearest')  # MATLAB imfilter 'replicate' (correlation)


def decimate(img, ratio):
    phase = int(np.floor((ratio + 1) / 2 + 0.5)) - 1  # MATLAB imresize 'nearest' sample position
    return img[phase::ratio, phase::ratio]


H_MS = [kernel(g, 4) for g in GNYQ_MS]
H_LPG = kernel(0.3, 4, toolbox_lpfilter=True)
H_PAN = {r: kernel(GNYQ_PAN, r) for r in (2, 4)}


def mtf_glp(lms, pan):
    """MTF_GLP.m: I_MS + imageHR - PAN_LP with per-band PAN histogram matching."""
    out = np.empty_like(lms)
    pan_std_lp = np.std(imfilter(pan, H_LPG), ddof=1)
    for b in range(lms.shape[0]):
        eq = (pan - pan.mean()) * (np.std(lms[b], ddof=1) / pan_std_lp) + lms[b].mean()
        pan_lp = _interp23tap(decimate(imfilter(eq, H_MS[b]), 4), 4)
        out[b] = lms[b] + eq - pan_lp
    return out


def lowpass(pan, ratio):
    return _interp23tap(decimate(imfilter(pan, H_PAN[ratio]), ratio), ratio)


def ceiling(err, bases, cell):
    """Fraction of err energy removed by per-band, per-cell least squares on bases (shared across bands)."""
    n, nb, h, w = err.shape; k = bases.shape[1]; g = h // cell
    B = bases.reshape(n, k, g, cell, g, cell).transpose(0, 2, 4, 3, 5, 1).reshape(n, g, g, cell * cell, k)
    ata = np.einsum('ngpik,ngpil->ngpkl', B, B) + 1e-12 * np.eye(k)
    resid = 0.0
    for b in range(nb):
        e = err[:, b].reshape(n, g, cell, g, cell).transpose(0, 1, 3, 2, 4).reshape(n, g, g, cell * cell)
        coef = np.linalg.solve(ata, np.einsum('ngpik,ngpi->ngpk', B, e)[..., None])[..., 0]
        resid += float(((e - np.einsum('ngpik,ngpk->ngpi', B, coef)) ** 2).sum())
    return 1 - resid / float((err ** 2).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    env = {'python': sys.version, 'numpy': np.__version__, 'source_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.json')}

    print('phase: Ga MTF-GLP vs A0', flush=True)
    ga = {}
    for protocol, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        path = list(a.data.rglob(filename)); assert len(path) == 1
        a0 = json.loads((A0 / f'{protocol}.json').read_text())
        if a0['data_sha256'] != sha(path[0]): raise ValueError('A0 evaluation used different data')
        rows = []
        with h5py.File(path[0], 'r') as f:
            for i in range(len(f['pan'])):
                pan = np.asarray(f['pan'][i], dtype=np.float64)[0]; lms = np.asarray(f['lms'][i], dtype=np.float64); ms = np.asarray(f['ms'][i], dtype=np.float64)
                fused = mtf_glp(lms, pan)
                m = rr_metrics(np.asarray(f['gt'][i], dtype=np.float64), fused) if protocol == 'RR' else fr_metrics(fused, lms, ms, pan)
                rows.append({'index': i, 'metrics': m})
        mean = {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
        a0_rows = a0['rows']
        better = {'PSNR': 1, 'SCC': 1, 'Q2n': 1, 'QNR': 1, 'SAM': -1, 'ERGAS': -1, 'D_lambda': -1, 'D_s': -1}
        ga[protocol] = {'MTF_GLP': mean, 'A0': a0['mean'], 'delta_vs_A0': {k: mean[k] - a0['mean'][k] for k in mean},
                        'scenes_better_than_A0': {k: int(sum(better[k] * (r['metrics'][k] - q['metrics'][k]) > 0 for r, q in zip(rows, a0_rows))) for k in mean},
                        'rows': rows, 'data_sha256': a0['data_sha256']}
        print(protocol, json.dumps(ga[protocol]['delta_vs_A0']), flush=True)
    ga_pass = ga['FR']['MTF_GLP']['D_s'] <= ga['FR']['A0']['D_s'] - 0.005
    write(a.output / 'Ga.json', {'environment': env, 'results': ga, 'gate': {'rule': CONFIG['Ga']['pass'], 'pass': ga_pass}})

    print('phase: Gb oracle ceiling', flush=True)
    found = list(a.data.rglob('valid_qb.h5')); assert len(found) == 1
    with h5py.File(found[0], 'r') as f:
        pan = np.asarray(f['pan'][:], dtype=np.float64)[:, 0] / 2047; lms = np.asarray(f['lms'][:], dtype=np.float64) / 2047; gt = np.asarray(f['gt'][:], dtype=np.float64) / 2047
    lp2 = np.stack([lowpass(p, 2) for p in pan]); lp4 = np.stack([lowpass(p, 4) for p in pan])
    d1, d2 = pan - lp2, lp2 - lp4
    err = gt - lms
    e_lms = float((err ** 2).mean())
    gb = {'E_LMS': e_lms, 'A0_explained': 1 - A0_VALIDATION['E_A'] / e_lms, 'A0_source': A0_VALIDATION, 'fits': {}}
    for label, bases in (('K1', (d1 + d2)[:, None]), ('K2', np.stack([d1, d2], 1))):
        null = np.roll(bases, 1, axis=0)
        for cell in (4, 8):
            real, chance = ceiling(err, bases, cell), ceiling(err, null, cell)
            gb['fits'][f'{label}_cell{cell}'] = {'oracle': real, 'null': chance, 'oracle_minus_null': real - chance}
            print(label, cell, real, chance, flush=True)
    gb_pass = gb['fits']['K2_cell4']['oracle_minus_null'] >= gb['A0_explained']
    write(a.output / 'Gb.json', {'environment': env, 'validation_sha256': sha(found[0]), 'results': gb, 'gate': {'rule': CONFIG['Gb']['pass'], 'pass': gb_pass}})
    write(a.output / 'complete.json', {'Ga_pass': ga_pass, 'Gb_pass': gb_pass, 'training_started': False, 'optimizer_updates': 0, 'time': time.time()})
    print('gates', ga_pass, gb_pass, flush=True)


if __name__ == '__main__':
    main()
