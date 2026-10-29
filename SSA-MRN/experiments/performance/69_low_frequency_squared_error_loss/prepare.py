"""Experiment 69 sensor operators and preparation.

Module level (imported by the 58 trainer as `prepare`, like 65/67): the 54 sensor table, Wald operators and 2x coarser
pairs unchanged, plus the two helpers of the 69 losses: decimate() at one phase per patch and find_phases().

CLI (no training, local CPU), the facts the loss design rests on, fixed before training:
--check: on 256 evenly spaced QB training patches, whether the given MS equals D(GT) (54 QB Wald operator: MS MTF,
         41 taps, replicate border, x4 decimation) at the fixed phase 2 and at the best of the 16 phases per patch, inside
         the 5-pixel LR margin, and the same for the 2x coarser pairs (54 CoarsePairs, phase 2 by construction). The training
         patches carry flips/rotations, so the decimation phase differs per patch; with the patch's own phase MS = D(GT).
--calibrate: loss weights by the 65 rule (same relative strength at a converged output). At the archived 65 C2MS_L1S
         QB seed 42 output y on the same patches (8 batches of 32), gradient norms with respect to y of
         L1_LF = mean|L(y - g)|, MSE_LF = mean (L(y - g))^2 and OBS = mean (D_p(y) - D_p(g))^2 over the whole LR grid with
         the patch's own phase p (equal to (D_p(y) - MS)^2 inside the margin, border-safe), where L is the per-band
         MS MTF blur at full resolution (60/68). lambda_O = |grad L1_LF| / |grad OBS|, k_S = |grad L1_LF| / |grad MSE_LF|
         (batch means). The 53 C2MS output is recorded for reference. Writes provenance/*.json only.
The CLI uses the FFT form of the blur (68 fft_blur, equal to the direct form within 5e-12 DN) because the direct 41x41
float64 convolution does not fit the local CPU memory; the trainer and run.py use the direct 54 form on the GPU.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]; PERF = ROOT / 'experiments/performance'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


P54 = _module('exp54_prepare', PERF / '54_multi_sensor_scale_mixed_training/prepare.py')
SENSORS, PHASE, SIZE = P54.SENSORS, P54.PHASE, P54.SIZE
data_files, gen_mtf, Wald, CoarsePairs, _blur = P54.data_files, P54.gen_mtf, P54.Wald, P54.CoarsePairs, P54._blur
MARGIN = 5


def decimate(b, phase=None):
    """x4 decimation of (N,C,H,W): phase 2 (the 54 Wald operator, None) or one phase per sample (0-15 = row * 4 + column)."""
    if phase is None: return b[..., PHASE::4, PHASE::4]
    n, c, h, w = b.shape
    grid = b.reshape(n, c, h // 4, 4, w // 4, 4).permute(0, 3, 5, 1, 2, 4).reshape(n, 16, c, h // 4, w // 4)
    return grid[torch.arange(n, device=b.device), torch.as_tensor(np.asarray(phase), device=b.device, dtype=torch.long)]


def find_phases(blurred_gt, ms, margin=MARGIN):
    """Per-patch phase: the one of 16 whose decimation of the MS-MTF-blurred GT has the smallest interior mismatch to the
    given MS; returns (phase, interior RMS of the mismatch at that phase) in the input units."""
    n = len(ms)
    r = torch.stack([(decimate(blurred_gt, np.full(n, q)) - ms)[..., margin:-margin, margin:-margin].square().mean((1, 2, 3)) for q in range(16)], 1)
    return r.argmin(1), r.min(1).values.sqrt()


# ---------------------------------------------------------------- CLI (local CPU)
CHECKPOINTS = {'L1S_65': PERF / '65_l1_spectral_loss/checkpoints/C2MS_L1S_QB_s42/best.pt',
               'C2MS_53': PERF / '53_scale_mixed_training/checkpoints/C2MS_QB_s42/best.pt'}
TRAIN = ROOT / 'data/train/train_qb.h5'; TRAIN_SHA = '7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371'
PATCHES, BATCH = 256, 32


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def patches():
    import h5py
    if sha(TRAIN) != TRAIN_SHA: raise ValueError('train_qb.h5 sha mismatch')
    with h5py.File(TRAIN, 'r') as f:
        idx = np.linspace(0, len(f['pan']) - 1, PATCHES).astype(int)
        return idx, {k: torch.from_numpy(np.asarray(f[k][np.sort(idx)], dtype=np.float64) / 2047.) for k in ('pan', 'lms', 'ms', 'gt')}


def operators(sensor='QB'):
    """D(x, phase): MS MTF blur and x4 decimation (phase 2 or one phase per patch); L: the low band (same MS MTF blur)."""
    grad = torch.is_grad_enabled()
    fft_blur = _module('exp68_prepare', PERF / '68_ssai_folding_sam_split_seed_ensemble/prepare.py').fft_blur
    torch.set_grad_enabled(grad)  # the 68 module disables gradients when it is loaded
    w = Wald(sensor, 'cpu'); low = torch.from_numpy(np.stack([gen_mtf(g, 4) for g in SENSORS[sensor]['ms_gnyq']])[:, None]).double()
    B = lambda x: fft_blur(x, w.ms_kernel)
    D = lambda x, phase=None: decimate(B(x), phase)
    L = lambda x: fft_blur(x, low)
    return D, L, B


def phases(gt, ms, B):
    """Per-patch phase for the CLI: find_phases on the FFT-blurred GT; RMS in DN."""
    ph, rms = find_phases(B(gt), ms)
    return ph, rms * 2047


def mask(shape, margin=MARGIN):
    m = torch.zeros(shape, dtype=torch.float64); m[..., margin:-margin, margin:-margin] = 1; return m


def check(out):
    idx, a = patches(); D, _, B = operators()
    e = (D(a['gt']) - a['ms']) * 2047; m = mask(e.shape); ph, rms = phases(a['gt'], a['ms'], B)
    ep = (D(a['gt'], ph) - a['ms']) * 2047
    res = dict(patches=PATCHES, indices='np.linspace(0, n-1, 256).astype(int) over train_qb.h5',
               original_phase2=dict(full_rms_DN=float(e.square().mean().sqrt()), interior_max_abs_DN=float((e * m).abs().max()),
                                    interior_rms_DN=float(((e * m).square().sum() / m.sum()).sqrt()), ms_rms_DN=float((a['ms'] * 2047).square().mean().sqrt())),
               original_patch_phase=dict(interior_max_abs_DN=float((ep * m).abs().max()), interior_rms_max_over_patches_DN=float(rms.max()),
                                         full_rms_DN=float(ep.square().mean().sqrt()), border_max_abs_DN=float((ep * (1 - m)).abs().max()),
                                         phase_counts={f'({q // 4},{q % 4})': int(c) for q, c in enumerate(np.bincount(ph.numpy(), minlength=16)) if c},
                                         interior_pixels_per_band=int(m[0, 0].sum())))
    cp = CoarsePairs('QB', 'cpu'); e2 = []
    for s in range(0, PATCHES, 16):  # 54 CoarsePairs uses the direct 41x41 convolution; small batches keep its im2col buffer in memory
        _, _, ms2, gt2 = cp(a['pan'][s:s + 16].float(), a['gt'][s:s + 16].float())
        e2.append((D(gt2.double()) - ms2.double()) * 2047)
    e2 = torch.cat(e2)
    res['coarse'] = dict(full_max_abs_DN=float(e2.abs().max()), shape_ms=list(ms2.shape[-2:]), note='coarse MS is generated by the same Wald operator from the coarse GT')
    (out / 'observation_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


def calibrate(out):
    idx, a = patches(); D, L, B = operators(); res = dict(patches=PATCHES, batches=PATCHES // BATCH, margin=MARGIN, models={})
    for name, path in CHECKPOINTS.items():
        net = _module('exp54_model', PERF / '54_multi_sensor_scale_mixed_training/model.py').Model('C2MS', 4, 42); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True); net.eval()
        norms = {k: [] for k in ('L1_all', 'L1_LF', 'MSE_LF', 'OBS', 'MSE_all')}
        for s in range(0, PATCHES, BATCH):
            b = {k: v[s:s + BATCH] for k, v in a.items()}
            with torch.no_grad(): y0 = net(*(b[k].float() for k in ('pan', 'lms', 'ms'))).double()
            g = b['gt']; ph, _ = phases(g, b['ms'], B); dg = D(g, ph)
            terms = {'L1_all': lambda y: (y - g).abs().mean(), 'L1_LF': lambda y: L(y - g).abs().mean(),
                     'MSE_LF': lambda y: L(y - g).square().mean(), 'MSE_all': lambda y: (y - g).square().mean(),
                     'OBS': lambda y: (D(y, ph) - dg).square().mean()}
            for k, f in terms.items():
                y = y0.clone().requires_grad_(True)
                (grad,) = torch.autograd.grad(f(y), y); norms[k].append(float(grad.norm()))
        mean = {k: float(np.mean(v)) for k, v in norms.items()}
        res['models'][name] = dict(checkpoint=str(path.relative_to(ROOT)), sha256=sha(path), grad_norm_mean=mean,
                                   per_batch_lambda_O=[l / o for l, o in zip(norms['L1_LF'], norms['OBS'])],
                                   per_batch_k_S=[l / q for l, q in zip(norms['L1_LF'], norms['MSE_LF'])],
                                   lambda_O=mean['L1_LF'] / mean['OBS'], k_S=mean['L1_LF'] / mean['MSE_LF'])
        print(name, json.dumps(res['models'][name]['grad_norm_mean']), 'lambda_O', res['models'][name]['lambda_O'], 'k_S', res['models'][name]['k_S'])
    res['rule'] = 'lambda_O and k_S are taken from L1S_65 (the recipe being changed); C2MS_53 is recorded for reference only'
    res['lambda_O'] = round(res['models']['L1S_65']['lambda_O'], 2); res['k_S'] = round(res['models']['L1S_65']['k_S'], 2)
    (out / 'loss_weight_calibration.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


def main():
    import argparse, os
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); p.add_argument('--calibrate', action='store_true')
    a = p.parse_args(); out = HERE / 'provenance'; out.mkdir(parents=True, exist_ok=True); torch.set_grad_enabled(True)
    if a.check: check(out)
    if a.calibrate: calibrate(out)


if __name__ == '__main__':
    main()
