"""75 follow-up (no training): where do the gain of C2MS_W64D8 over the original and its loss against C2MS sit, by frequency band
and by structure size? Reuses the experiment-72 decomposition (LMDI over radial FFT bins, SAM Shapley, GT-only structure classes).
QB RR test, seed 42. Writes only to the scratchpad."""
import os, sys, json, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import numpy as np, torch

REPO = Path('/Users/biyonghiyon_e/Desktop/CtrS/SSA-MRN'); PERF = REPO / 'experiments/performance'
SCR = Path(__file__).resolve().parent; CACHE = SCR / 'cache75'; CACHE.mkdir(exist_ok=True); OUT = SCR / 'analysis75'; OUT.mkdir(exist_ok=True)
C72 = SCR.parent / 'cache72'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


P = module('p72', PERF / '72_c2ms_gain_frequency_structure_analysis/prepare.py')
M75 = module('m75', PERF / '75_parallel_branch_width_depth/model.py')
P.SEEDS = (42,); P.CACHE = CACHE
log = lambda *a: print(*a, flush=True)

# ---- predictions: C0 and C2MS from the 72 cache, the two 75 arms computed here; all checked against the 75 server evaluation rows
for arm in ('C0', 'C2MS'):
    dst = CACHE / f'QB_{arm}_s42.npy'
    if not dst.exists(): dst.symlink_to(C72 / f'QB_{arm}_s42.npy')
data = P.read_test('QB'); gt = data['gt']; n = len(gt)
ARMS = {'W64D8': 'C2MS_W64D8', 'W128D4': 'C2MS_W128D4'}
for short, arm in ARMS.items():
    dst = CACHE / f'QB_{short}_s42.npy'
    if dst.exists(): continue
    net = M75.Model(arm, 4, 42); ck = torch.load(PERF / f'75_parallel_branch_width_depth/checkpoints/{arm}_QB_s42/best.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(ck['model'], strict=True); net.eval()
    with torch.no_grad():
        ys = np.stack([P.P68.predict(net, (data['pan'][i], data['lms'][i], data['ms'][i]), 2047.) for i in range(n)])
    np.save(dst, ys.astype(np.float32)); log('predicted', arm)
check = {}
for short, rows_arm in (('C0', 'C0'), ('C2MS', 'C2MS'), ('W64D8', 'C2MS_W64D8'), ('W128D4', 'C2MS_W128D4')):
    src = 'C2MS_W128D4' if short == 'W128D4' else 'C2MS_W64D8'
    rows = json.load(open(PERF / f'75_parallel_branch_width_depth/results/{src}_QB_s42/evaluation/QB_v2_4.json'))['rows'][rows_arm]
    y = P.cached('QB', short, 42); psnr = [P.rr_metrics(gt[i], y[i])['PSNR'] for i in range(n)]
    check[short] = dict(mean_psnr=float(np.mean(psnr)), max_abs_diff_vs_server=float(max(abs(p - r['metrics']['RR']['PSNR']) for p, r in zip(psnr, rows))))
log('reproduction', check)
if max(v['max_abs_diff_vs_server'] for v in check.values()) > 0.01: raise SystemExit('predictions do not reproduce the server evaluation')

ARMLIST = ('C0', 'W64D8', 'W128D4', 'C2MS')
PAIRS = (('C0', 'W64D8'), ('W64D8', 'C2MS'), ('C0', 'C2MS'), ('C0', 'W128D4'), ('W128D4', 'C2MS'), ('W128D4', 'W64D8'))
masks = P.bin_masks(*gt.shape[-2:])
freq = P.frequency_block('QB', gt, data['lms'], ARMLIST, PAIRS, masks)
log('frequency done')
sam = {f'{a}->{b}': P.sam_shapley('QB', gt, a, b, masks) for a, b in (('C0', 'W64D8'), ('W64D8', 'C2MS'), ('C0', 'C2MS'))}
log('sam done')
I = gt.mean(1); Ms = np.stack([P.scale_maps(i) for i in I]); gate = float(np.percentile(Ms[:, 0], P.GATE))
cl = np.stack([P.classify(I[i], gate, M=Ms[i]) for i in range(n)]); errs = P.error_stack('QB', gt, ARMLIST)
struct = P.class_table(errs, cl, P.NAMES, PAIRS)
# class x band (as 72 run_structure): error energy and detail gain y.g/g^2 per structure class inside four frequency bands
h, w = gt.shape[-2:]; r = np.hypot(*np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij'))
parts = dict(zip(('>16px', '8-16px', '4-8px', '<4px'), (r < 1 / 16, (r >= 1 / 16) & (r < 1 / 8), (r >= 1 / 8) & (r < 1 / 4), r >= 1 / 4)))
Fg = np.fft.fft2(gt.astype(np.float32), axes=(-2, -1)); X = {}
for arm in ARMLIST:
    Fe = np.fft.fft2(errs[arm], axes=(-2, -1)); Fy = Fe + Fg; X[arm] = {}
    for p, m in parts.items():
        e = np.fft.ifft2(Fe * m, axes=(-2, -1)).real; y = np.fft.ifft2(Fy * m, axes=(-2, -1)).real; g = np.fft.ifft2(Fg * m, axes=(-2, -1)).real; row = {}
        for k, name in enumerate(P.NAMES):
            mask = np.broadcast_to((cl == k)[:, None], g.shape)
            row[name] = dict(mse=float((e ** 2).mean(0)[mask].mean()), gain=float((y * g[None]).mean(0)[mask].sum() / max((g ** 2)[mask].sum(), 1e-9)))
        X[arm][p] = row
json.dump(dict(reproduction=check, bins=P.BINS, frequency=freq, sam_shapley=sam, structure=struct, class_by_frequency=X, gate=gate),
          open(OUT / 'analysis75.json', 'w'), indent=1, ensure_ascii=False)
log('done', OUT / 'analysis75.json')
