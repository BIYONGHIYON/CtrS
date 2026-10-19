"""Experiment 46 (no training): reduced-resolution validity check of the FR indices.

RR test inputs are scored with the FR indices exactly as FR inputs, for outputs whose fidelity to GT is known
(GT itself, LMS, MTF-GLP, experiment 43 models with and without the experiment 45 projection, and blends).
Tests and thresholds are fixed in config.json before measurement.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/performance'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(ROOT))
from common.metrics import rr_metrics, fr_metrics, q2n
from common.observation import SensorObservation


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


P45 = load('exp45', EXP / '45_exact_observation_projection/prepare.py')   # Projector and experiment 43 Model
P36 = load('exp36', EXP / '36_gain_map_detail_injection/prepare.py')      # mtf_glp
BLUR = SensorObservation('QB', 'decimate', (2, 2)).double()
BETTER = {'MSE': -1, 'D_lambda': -1, 'D_s': -1, 'QNR': 1, 'D_lambda_K': -1, 'HQNR': 1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def indices(fused, gt, lms, ms, pan):
    fr = fr_metrics(fused, lms, ms, pan)
    with torch.no_grad():
        blurred = BLUR.blur(torch.from_numpy(fused).unsqueeze(0))[0].numpy()
    dlk = 1 - q2n(lms, blurred)
    rr = rr_metrics(gt, fused)
    return {'MSE': float(np.mean((fused - gt) ** 2)), 'PSNR': rr['PSNR'], 'SAM': rr['SAM'],
            'D_lambda': fr['D_lambda'], 'D_s': fr['D_s'], 'QNR': fr['QNR'], 'D_lambda_K': dlk, 'HQNR': (1 - dlk) * (1 - fr['D_s'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='local smoke test only')
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
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
            nets[f'{arm}_s{seed}'] = (net.to(a.device).eval(), sha(ck))
    scenes = []
    with h5py.File(found[0], 'r') as f:
        for i in range(n):
            arr = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
            x = [torch.from_numpy((arr[k] / 2047).astype('float32')).unsqueeze(0).to(a.device) for k in ('pan', 'lms', 'ms')]
            out = {'GT': arr['gt'], 'LMS': arr['lms'], 'MTF_GLP': P36.mtf_glp(arr['lms'], arr['pan'][0])}
            for key, (net, _) in nets.items():
                with torch.no_grad():
                    y = net(*x).float()
                yp, _ = proj.project(y, x[2], 5)
                out[key] = y[0].cpu().numpy().astype(np.float64) * 2047
                out[key + '_P5'] = yp[0].detach().cpu().numpy().astype(np.float64) * 2047
            for key in ('C0_s42', 'C2_s42'):
                if key not in out: continue
                for t in (0.25, 0.5, 0.75):
                    out[f'{key}_toGT{t}'] = (1 - t) * out[key] + t * arr['gt']
                for t in (0.25, 0.5):
                    out[f'{key}_toLMS{t}'] = (1 - t) * out[key] + t * arr['lms']
            rows = {k: indices(v, arr['gt'], arr['lms'], arr['ms'], arr['pan']) for k, v in out.items()}
            scenes.append({'index': i, 'candidates': rows})
            print(i, 'GT D_s %.4f  C0 D_s %.4f  LMS D_s %.4f' % (rows['GT']['D_s'], rows.get('C0_s42', rows['GT'])['D_s'], rows['LMS']['D_s']), flush=True)

    names = list(scenes[0]['candidates'])
    c0 = [k for k in names if k.startswith('C0_s') and k.count('_') == 1]
    tests = {}
    for idx in ('D_s', 'D_lambda', 'QNR', 'D_lambda_K', 'HQNR'):
        sign = BETTER[idx]
        t1 = sum(sign * (s['candidates']['GT'][idx] - np.mean([s['candidates'][k][idx] for k in c0])) > 0 for s in scenes)
        rhos = [spearmanr([s['candidates'][k]['MSE'] for k in names], [s['candidates'][k][idx] for k in names])[0] for s in scenes]
        # Positive rho means "more error -> worse index" after aligning direction with BETTER.
        rhos = [r * (-sign) for r in rhos]
        t3 = {}
        for key in ('C0_s42', 'C2_s42'):
            if f'{key}_toGT0.75' in names:
                t3[key] = int(sum(sign * (s['candidates'][f'{key}_toGT0.75'][idx] - s['candidates'][key][idx]) > 0 for s in scenes))
        tests[idx] = {'T1_GT_better_than_C0_mean': int(t1), 'T2_median_rank_corr_error_vs_badness': float(np.median(rhos)), 'T2_per_scene': [float(r) for r in rhos], 'T3_blend_to_GT_better': t3}
    t = tests['D_s']; m = len(scenes)
    if t['T1_GT_better_than_C0_mean'] <= m // 4 and t['T2_median_rank_corr_error_vs_badness'] <= -0.5: decision = 'conflicts'
    elif t['T1_GT_better_than_C0_mean'] >= m * 3 // 4 and t['T2_median_rank_corr_error_vs_badness'] >= 0.5: decision = 'consistent'
    else: decision = 'mixed'
    means = {k: {idx: float(np.mean([s['candidates'][k][idx] for s in scenes])) for idx in scenes[0]['candidates'][k]} for k in names}
    env = {'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
           'source_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.json'), 'data_sha256': sha(found[0]),
           'checkpoints': {k: v[1] for k, v in nets.items()}}
    write(a.output / 'validity.json', {'environment': env, 'tests': tests, 'decision_D_s': decision, 'means': means, 'scenes': scenes})
    write(a.output / 'complete.json', {'decision_D_s': decision, 'training_started': False, 'time': time.time(), 'limit': a.limit})
    print('decision', decision, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'T2_per_scene'} for k, v in tests.items()}), flush=True)


if __name__ == '__main__':
    main()
