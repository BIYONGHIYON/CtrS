"""77 follow-up (no training): where did the enlarged full-resolution backbone change the error?
E1 = C2MS_B128D8 and E2 = C2MS_B64D8 against 53 C2MS, E3 = C2MS_B64D8_W64D8 against 75 C2MS_W64D8 and 53 C2MS, by frequency band,
by structure size. Reuses the experiment-72 decomposition (LMDI over radial FFT bins, SAM Shapley, GT-only structure classes);
border77.py adds the 76 border/interior split. QB RR test, seed 42. Writes only next to this script."""
import os, sys, json, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import numpy as np, torch

REPO = Path('/Users/biyonghiyon_e/Desktop/CtrS/SSA-MRN'); PERF = REPO / 'experiments/performance'
SCR = Path(__file__).resolve().parent; CACHE = SCR / 'cache77'; CACHE.mkdir(exist_ok=True); OUT = SCR / 'analysis77'; OUT.mkdir(exist_ok=True)
X = PERF / '77_full_resolution_backbone_capacity'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


P = module('p72', PERF / '72_c2ms_gain_frequency_structure_analysis/prepare.py')
M75 = module('m75', PERF / '75_parallel_branch_width_depth/model.py')
M77 = module('m77', X / 'model.py')
P.SEEDS = (42,); P.CACHE = CACHE
log = lambda *a: print(*a, flush=True)
data = P.read_test('QB'); gt = data['gt']; n = len(gt)
ARMS = {'E1': 'C2MS_B128D8', 'E2': 'C2MS_B64D8', 'E3': 'C2MS_B64D8_W64D8'}


def net_of(short):
    if short in ('C0', 'C2MS'): return P.load_net('QB', short, 42)
    if short == 'W64D8':
        net = M75.Model('C2MS_W64D8', 4, 42); path = PERF / '75_parallel_branch_width_depth/checkpoints/C2MS_W64D8_QB_s42/best.pt'
    else:
        net = M77.Model(ARMS[short], 4, 42); path = X / f'checkpoints/{ARMS[short]}_QB_s42/best.pt'
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True); return net.eval()


ARMLIST = ('C0', 'W64D8', 'E3', 'C2MS', 'E2', 'E1')
for short in ARMLIST:
    dst = CACHE / f'QB_{short}_s42.npy'
    if dst.exists(): continue
    net = net_of(short)
    with torch.no_grad():
        ys = np.stack([P.P68.predict(net, (data['pan'][i], data['lms'][i], data['ms'][i]), 2047.) for i in range(n)])
    np.save(dst, ys.astype(np.float32)); log('predicted', short)
E = {a: json.load(open(X / f'results/{a}_QB_s42/evaluation/QB_v2_4.json'))['rows'] for a in ARMS.values()}
R75 = json.load(open(PERF / '75_parallel_branch_width_depth/results/C2MS_W64D8_QB_s42/evaluation/QB_v2_4.json'))['rows']
server = {'C0': E['C2MS_B64D8']['C0'], 'C2MS': E['C2MS_B64D8']['C2MS'], 'W64D8': R75['C2MS_W64D8'], **{s: E[a][a] for s, a in ARMS.items()}}
check = {}
for short, rows in server.items():
    y = P.cached('QB', short, 42); psnr = [P.rr_metrics(gt[i], y[i])['PSNR'] for i in range(n)]
    check[short] = dict(mean_psnr=float(np.mean(psnr)), max_abs_diff_vs_server=float(max(abs(p - r['metrics']['RR']['PSNR']) for p, r in zip(psnr, rows))))
log('reproduction', check)
if max(v['max_abs_diff_vs_server'] for v in check.values()) > 0.01: raise SystemExit('predictions do not reproduce the server evaluation')

PAIRS = (('C2MS', 'E1'), ('C2MS', 'E2'), ('W64D8', 'E3'), ('C2MS', 'E3'), ('C0', 'C2MS'))
masks = P.bin_masks(*gt.shape[-2:])
freq = P.frequency_block('QB', gt, data['lms'], ARMLIST, PAIRS, masks)
log('frequency done')
sam = {f'{a}->{b}': P.sam_shapley('QB', gt, a, b, masks) for a, b in PAIRS[:4]}
log('sam done')
I = gt.mean(1); Ms = np.stack([P.scale_maps(i) for i in I]); gate = float(np.percentile(Ms[:, 0], P.GATE))
cl = np.stack([P.classify(I[i], gate, M=Ms[i]) for i in range(n)]); errs = P.error_stack('QB', gt, ARMLIST)
struct = P.class_table(errs, cl, P.NAMES, PAIRS)
h, w = gt.shape[-2:]; r = np.hypot(*np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij'))
parts = dict(zip(('>16px', '8-16px', '4-8px', '<4px'), (r < 1 / 16, (r >= 1 / 16) & (r < 1 / 8), (r >= 1 / 8) & (r < 1 / 4), r >= 1 / 4)))
Fg = np.fft.fft2(gt.astype(np.float32), axes=(-2, -1)); XB = {}
for arm in ARMLIST:
    Fe = np.fft.fft2(errs[arm], axes=(-2, -1)); Fy = Fe + Fg; XB[arm] = {}
    for p, m in parts.items():
        e = np.fft.ifft2(Fe * m, axes=(-2, -1)).real; y = np.fft.ifft2(Fy * m, axes=(-2, -1)).real; g = np.fft.ifft2(Fg * m, axes=(-2, -1)).real; row = {}
        for k, name in enumerate(P.NAMES):
            mask = np.broadcast_to((cl == k)[:, None], g.shape)
            row[name] = dict(mse=float((e ** 2).mean(0)[mask].mean()), gain=float((y * g[None]).mean(0)[mask].sum() / max((g ** 2)[mask].sum(), 1e-9)))
        XB[arm][p] = row
json.dump(dict(reproduction=check, bins=P.BINS, frequency=freq, sam_shapley=sam, structure=struct, class_by_frequency=XB, gate=gate),
          open(OUT / 'frequency_structure.json', 'w'), indent=1, ensure_ascii=False)
log('done', OUT)
