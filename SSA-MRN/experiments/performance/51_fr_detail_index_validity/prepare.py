"""Experiment 51 (no training): ground-truth ranking test (experiment 46) for reference-free spatial-detail indices.

RR test inputs are treated as FR inputs; candidates of known fidelity are scored with each new index.
Indices, tests and thresholds are fixed in config.json before measurement.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch.nn import functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/performance'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(ROOT))
from common.metrics import fr_metrics  # noqa: E402
from common import evaluation_protocol as protocol  # noqa: E402

INDICES = ('PDC_corr', 'PDC_nrmse', 'BandHP_corr', 'D_rho_style')
MARGIN = 8
RATIO = 4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


P45 = load('exp45', EXP / '45_exact_observation_projection/prepare.py')   # Projector and experiment 43 Model
P36 = load('exp36', EXP / '36_gain_map_detail_injection/prepare.py')      # mtf_glp
M48 = load('exp48_model', EXP / '48_parallel_capacity_observation_refinement/model.py')  # F1 (experiment 49)
PAN_KERNEL = torch.from_numpy(protocol._pan_kernel(protocol.PAN_GNYQ['QB'])[None, None])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def pan_blur(x):
    """PAN-MTF blur with replicate padding for a (C,H,W) float64 array (no sampling)."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float64))[:, None]
    pad = PAN_KERNEL.shape[-1] // 2
    rows = torch.arange(-pad, t.shape[-2] + pad).clamp(0, t.shape[-2] - 1)
    cols = torch.arange(-pad, t.shape[-1] + pad).clamp(0, t.shape[-1] - 1)
    return F.conv2d(t.index_select(-2, rows).index_select(-1, cols), PAN_KERNEL)[:, 0].numpy()


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def local_corr(a, b, w=RATIO):
    """Sliding w x w Pearson correlation map (valid windows)."""
    ta, tb = torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None]
    m = lambda x: F.avg_pool2d(x, w, stride=1)
    ma, mb = m(ta), m(tb)
    cov = m(ta * tb) - ma * mb
    va, vb = (m(ta * ta) - ma * ma).clamp_min(0), (m(tb * tb) - mb * mb).clamp_min(0)
    return (cov / (va * vb).sqrt().clamp_min(1e-12)).numpy()[0, 0]


class DetailIndices:
    """Per-scene setup from observed PAN and MS only (no GT)."""

    def __init__(self, pan, ms):
        self.pan = np.asarray(pan, dtype=np.float64)[0]
        p_lr = pan_blur(self.pan[None])[0][2::RATIO, 2::RATIO]
        A = np.concatenate([ms.reshape(ms.shape[0], -1).T, np.ones((p_lr.size, 1))], 1)
        coef, *_ = np.linalg.lstsq(A, p_lr.reshape(-1), rcond=None)
        self.w, self.c = coef[:-1], coef[-1]
        self.hp_pan = (self.pan - pan_blur(self.pan[None])[0])
        s = (slice(MARGIN, -MARGIN), slice(MARGIN, -MARGIN))
        self.s = s

    def __call__(self, fused):
        fused = np.asarray(fused, dtype=np.float64)
        s, hp_pan = self.s, self.hp_pan
        i_f = np.tensordot(self.w, fused, 1) + self.c
        hp_i = i_f - pan_blur(i_f[None])[0]
        hp_bands = fused - pan_blur(fused)
        rho = np.mean([1 - local_corr(fused[b], self.pan) for b in range(fused.shape[0])])
        return {'PDC_corr': 1 - corr(hp_i[s], hp_pan[s]),
                'PDC_nrmse': float(np.sqrt(np.mean((hp_i[s] - hp_pan[s]) ** 2)) / np.sqrt(np.mean(hp_pan[s] ** 2))),
                'BandHP_corr': 1 - float(np.mean([corr(hp_bands[b][s], hp_pan[s]) for b in range(fused.shape[0])])),
                'D_rho_style': float(rho)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='experiment 46 results/validity.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='local smoke test only')
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    reference = json.loads(a.reference.read_text(encoding='utf-8'))
    found = list(a.data.rglob('test_qb_multiExm1.h5')); assert len(found) == 1
    n = a.limit or 20
    proj = P45.Projector(a.device)
    arms, seeds = ('C0', 'C1', 'C2', 'C3'), (42, 43, 44)
    if a.limit: arms, seeds = ('C0', 'C2'), (42,)
    nets = {}
    for arm in arms:
        for seed in seeds:
            ck = P45.E43 / f'checkpoints/{arm}_QB_s{seed}/best.pt'
            net = P45.M43.Model(arm, seed); net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'], strict=True)
            nets[f'{arm}_s{seed}'] = (net.to(a.device).eval(), sha(ck), True)
    for seed in seeds:
        ck = EXP / f'49_observation_refinement_seed_repeat/checkpoints/F1_QB_s{seed}/best.pt'
        net = M48.Model('F1', seed); net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'], strict=True)
        nets[f'F1_s{seed}'] = (net.to(a.device).eval(), sha(ck), False)
    scenes, repro = [], []
    with h5py.File(found[0], 'r') as f:
        for i in range(n):
            arr = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
            x = [torch.from_numpy((arr[k] / 2047).astype('float32')).unsqueeze(0).to(a.device) for k in ('pan', 'lms', 'ms')]
            out = {'GT': arr['gt'], 'LMS': arr['lms'], 'MTF_GLP': P36.mtf_glp(arr['lms'], arr['pan'][0])}
            for key, (net, _, project) in nets.items():
                with torch.no_grad():
                    y = net(*x).float()
                out[key] = y[0].cpu().numpy().astype(np.float64) * 2047
                if project:  # experiment 46 set: 43 models also with the 45 P5 projection
                    yp, _ = proj.project(y, x[2], 5)
                    out[key + '_P5'] = yp[0].detach().cpu().numpy().astype(np.float64) * 2047
            for key in ('C0_s42', 'C2_s42'):
                if key not in out: continue
                for t in (0.25, 0.5, 0.75):
                    out[f'{key}_toGT{t}'] = (1 - t) * out[key] + t * arr['gt']
                for t in (0.25, 0.5):
                    out[f'{key}_toLMS{t}'] = (1 - t) * out[key] + t * arr['lms']
            idx = DetailIndices(arr['pan'], arr['ms'])
            rows = {k: {'MSE': float(np.mean((v - arr['gt']) ** 2)), **idx(v)} for k, v in out.items()}
            for k in ('GT', 'LMS', 'C0_s42'):
                if k in out:
                    ds = fr_metrics(out[k], arr['lms'], arr['ms'], arr['pan'])['D_s']
                    repro.append(abs(ds - reference['scenes'][i]['candidates'][k]['D_s']))
            scenes.append({'index': i, 'pan_weights': [float(v) for v in idx.w], 'pan_bias': float(idx.c), 'candidates': rows})
            print(i, ' '.join('%s GT %.4f C0 %.4f LMS %.4f' % (k, rows['GT'][k], rows.get('C0_s42', rows['GT'])[k], rows['LMS'][k]) for k in INDICES), flush=True)
    names = list(scenes[0]['candidates'])
    c0 = [k for k in names if k.startswith('C0_s') and k.count('_') == 1]
    tests, m = {}, len(scenes)
    for name in INDICES:  # all indices: lower is better
        t1 = sum(s['candidates']['GT'][name] < np.mean([s['candidates'][k][name] for k in c0]) for s in scenes)
        rhos = [spearmanr([s['candidates'][k]['MSE'] for k in names], [s['candidates'][k][name] for k in names])[0] for s in scenes]
        t3 = {key: int(sum(s['candidates'][f'{key}_toGT0.75'][name] < s['candidates'][key][name] for s in scenes)) for key in ('C0_s42', 'C2_s42') if f'{key}_toGT0.75' in names}
        t4 = sum(min(names, key=lambda k: s['candidates'][k][name]) == 'GT' for s in scenes)
        t2 = float(np.median(rhos))
        if t1 >= m * 3 // 4 and t2 >= 0.5: decision = 'passes'
        elif t1 <= m // 4 and t2 <= -0.5: decision = 'conflicts'
        else: decision = 'mixed'
        tests[name] = {'T1_GT_better_than_C0_mean': int(t1), 'T2_median_rank_corr_error_vs_badness': t2, 'T2_per_scene': [float(r) for r in rhos],
                       'T3_blend_to_GT_better': t3, 'T4_GT_best_of_all': int(t4), 'decision': decision}
    means = {k: {name: float(np.mean([s['candidates'][k][name] for s in scenes])) for name in scenes[0]['candidates'][k]} for k in names}
    env = {'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
           'source_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.json'), 'data_sha256': sha(found[0]),
           'reference_sha256': sha(a.reference), 'reproduction_max_abs_D_s': float(max(repro)), 'checkpoints': {k: v[1] for k, v in nets.items()}}
    write(a.output / 'validity.json', {'environment': env, 'tests': tests, 'means': means, 'scenes': scenes})
    write(a.output / 'complete.json', {'decisions': {k: v['decision'] for k, v in tests.items()}, 'reproduction_max_abs_D_s': float(max(repro)), 'training_started': False, 'time': time.time(), 'limit': a.limit})
    print('reproduction max |dD_s|', max(repro), flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'T2_per_scene'} for k, v in tests.items()}), flush=True)


if __name__ == '__main__':
    main()
