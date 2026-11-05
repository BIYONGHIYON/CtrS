"""76 follow-up (no training): did the quarter-resolution branch change the 4-8 px uncorrelated error and the flat-region loss?
A = C2MS_W64D8_W4 against 75 C2MS_W64D8, B = C2MS_W4 against 53 C2MS, by frequency band and by structure size. Reuses the experiment-72 decomposition (LMDI over radial FFT bins, SAM Shapley, GT-only structure classes).
QB RR test, seed 42. Writes only to the scratchpad."""
import os, sys, json, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import numpy as np, torch

REPO = Path('/Users/biyonghiyon_e/Desktop/CtrS/SSA-MRN'); PERF = REPO / 'experiments/performance'
SCR = Path(__file__).resolve().parent; CACHE = SCR / 'cache76'; CACHE.mkdir(exist_ok=True); OUT = SCR / 'analysis76'; OUT.mkdir(exist_ok=True)
C72 = SCR.parent / 'cache72'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


P = module('p72', PERF / '72_c2ms_gain_frequency_structure_analysis/prepare.py')
M76 = module('m76', PERF / '76_quarter_resolution_branch/model.py')
P.SEEDS = (42,); P.CACHE = CACHE
log = lambda *a: print(*a, flush=True)

# ---- predictions: C0/C2MS from the 72 cache, 75 W64D8 from the 75 analysis cache, the two 76 arms computed here
C75 = SCR.parent / 'p75' / 'cache75'
for arm, src in (('C0', C72 / 'QB_C0_s42.npy'), ('C2MS', C72 / 'QB_C2MS_s42.npy'), ('W64D8', C75 / 'QB_W64D8_s42.npy')):
    dst = CACHE / f'QB_{arm}_s42.npy'
    if not dst.exists(): dst.symlink_to(src)
data = P.read_test('QB'); gt = data['gt']; n = len(gt)
ARMS = {'A': 'C2MS_W64D8_W4', 'B': 'C2MS_W4'}
for short, arm in ARMS.items():
    dst = CACHE / f'QB_{short}_s42.npy'
    if dst.exists(): continue
    net = M76.Model(arm, 4, 42); ck = torch.load(PERF / f'76_quarter_resolution_branch/checkpoints/{arm}_QB_s42/best.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(ck['model'], strict=True); net.eval()
    with torch.no_grad():
        ys = np.stack([P.P68.predict(net, (data['pan'][i], data['lms'][i], data['ms'][i]), 2047.) for i in range(n)])
    np.save(dst, ys.astype(np.float32)); log('predicted', arm)
check = {}
R76 = {a: json.load(open(PERF / f'76_quarter_resolution_branch/results/{a}_QB_s42/evaluation/QB_v2_4.json'))['rows'] for a in ARMS.values()}
R75 = json.load(open(PERF / '75_parallel_branch_width_depth/results/C2MS_W64D8_QB_s42/evaluation/QB_v2_4.json'))['rows']
for short, rows in (('C0', R76['C2MS_W4']['C0']), ('C2MS', R76['C2MS_W4']['C2MS']), ('W64D8', R75['C2MS_W64D8']), ('A', R76['C2MS_W64D8_W4']['C2MS_W64D8_W4']), ('B', R76['C2MS_W4']['C2MS_W4'])):
    y = P.cached('QB', short, 42); psnr = [P.rr_metrics(gt[i], y[i])['PSNR'] for i in range(n)]
    check[short] = dict(mean_psnr=float(np.mean(psnr)), max_abs_diff_vs_server=float(max(abs(p - r['metrics']['RR']['PSNR']) for p, r in zip(psnr, rows))))
log('reproduction', check)
if max(v['max_abs_diff_vs_server'] for v in check.values()) > 0.01: raise SystemExit('predictions do not reproduce the server evaluation')

ARMLIST = ('C0', 'W64D8', 'A', 'C2MS', 'B')
PAIRS = (('W64D8', 'A'), ('C2MS', 'B'), ('A', 'C2MS'), ('W64D8', 'C2MS'), ('C0', 'C2MS'))
masks = P.bin_masks(*gt.shape[-2:])
freq = P.frequency_block('QB', gt, data['lms'], ARMLIST, PAIRS, masks)
log('frequency done')
sam = {f'{a}->{b}': P.sam_shapley('QB', gt, a, b, masks) for a, b in (('W64D8', 'A'), ('C2MS', 'B'))}
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
          open(OUT / 'analysis76.json', 'w'), indent=1, ensure_ascii=False)
log('done', OUT / 'analysis76.json')
