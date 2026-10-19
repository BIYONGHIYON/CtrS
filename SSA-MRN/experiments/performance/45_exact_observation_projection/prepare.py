"""Experiment 45 phase 0 without training: exact observation-consistency projection at inference.

For each experiment 43 model (C0-C3, seeds 42-44) the RR20/FR20 predictions are evaluated as is and after
y' = y + D^T M^T (M D D^T M^T)^-1 M (MS - D y), with D the audited QB MTF observation (experiment 3).
Gate thresholds are fixed in config.json before measurement.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E43 = ROOT / 'experiments/performance/43_normalized_parallel_capacity'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))

sys.path.insert(0, str(ROOT))
from common.observation import SensorObservation
from common.metrics import rr_metrics, fr_metrics
# Experiment 43's model module; common.model resolves to the repository copy, whose hash equality with
# experiment 43's preserved source is recorded below.
spec = importlib.util.spec_from_file_location('exp43_model', E43 / 'model.py')
M43 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M43)

MARGINS = {'P5': 5, 'P0': 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def code_text(path):
    # Line endings and trailing blank lines differ between Windows-preserved and repository copies.
    return Path(path).read_bytes().replace(b'\r\n', b'\n').rstrip(b'\n')


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


class Projector:
    def __init__(self, device):
        self.D = SensorObservation('QB', 'decimate', (2, 2)).to(device).float()

    def forward(self, y):
        return self.D(y)

    def adjoint(self, v, shape):
        y = torch.zeros(shape, device=v.device, dtype=v.dtype, requires_grad=True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(self.D(y), y, grad_outputs=v)
        return g

    def mask(self, lr_shape, margin, device):
        m = torch.zeros(lr_shape, device=device)
        h, w = lr_shape[-2:]
        m[..., margin:h - margin, margin:w - margin] = 1
        return m

    def project(self, y, ms, margin, tol=1e-6, max_iter=300):
        """Minimum-norm correction so that the masked observation of y equals ms (CG per band)."""
        m = self.mask(ms.shape, margin, y.device)
        A = lambda lam: m * self.forward(self.adjoint(m * lam, y.shape))
        b = m * (ms - self.forward(y))
        lam = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum((-2, -1), keepdim=True); b2 = rs.clamp_min(1e-30)
        it = 0
        for it in range(1, max_iter + 1):
            Ap = A(p)
            alpha = rs / (p * Ap).sum((-2, -1), keepdim=True).clamp_min(1e-30)
            lam = lam + alpha * p; r = r - alpha * Ap
            rs_new = (r * r).sum((-2, -1), keepdim=True)
            if bool(((rs_new / b2).sqrt() < tol).all()): rs = rs_new; break
            p = r + (rs_new / rs) * p; rs = rs_new
        out = y + self.adjoint(m * lam, y.shape)
        before = float((m * (ms - self.forward(y))).square().sum() / m.sum().clamp_min(1) / ms.shape[1])
        after = float((m * (ms - self.forward(out))).square().sum() / m.sum().clamp_min(1) / ms.shape[1])
        return out, {'iterations': it, 'relative_residual': float((rs / b2).sqrt().max()), 'obs_rmse_before_dn': before ** .5 * 2047, 'obs_rmse_after_dn': after ** .5 * 2047}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='local smoke test only: first N scenes and C0/C2 seed42')
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    proj = Projector(a.device)
    env = {'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
           'source_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.json'), 'observation_sha256': sha(ROOT / 'common/observation.py'),
           'exp43_model_sha256': sha(E43 / 'model.py'),
           'exp43_common_model_equals_repository': code_text(E43 / 'provenance/source/common/model.py') == code_text(ROOT / 'common/model.py'),
           'exp43_metrics_equals_repository': code_text(E43 / 'provenance/source/common/metrics.py') == code_text(ROOT / 'common/metrics.py'),
           'exp43_network_equals_repository': code_text(E43 / 'provenance/source/references/upstream/network.py') == code_text(ROOT / 'references/upstream/network.py')}
    paths = {}
    for protocol, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename)); assert len(found) == 1; paths[protocol] = found[0]
    n = a.limit or 20

    print('phase: data consistency of the constraint (RR GT)', flush=True)
    with h5py.File(paths['RR'], 'r') as f, torch.no_grad():
        d = []
        for i in range(n):
            gt = torch.from_numpy(np.asarray(f['gt'][i], dtype=np.float32) / 2047).unsqueeze(0).to(a.device)
            ms = torch.from_numpy(np.asarray(f['ms'][i], dtype=np.float32) / 2047).unsqueeze(0).to(a.device)
            err = (proj.forward(gt) - ms) * 2047
            inner = proj.mask(ms.shape, 5, a.device).bool()
            d.append((float(err[inner].square().mean().sqrt()), float(err[~inner].square().mean().sqrt())))
    consistency = {'rr_gt_interior_rmse_dn_mean': float(np.mean([x[0] for x in d])), 'rr_gt_interior_rmse_dn_max': float(np.max([x[0] for x in d])),
                   'rr_gt_border_rmse_dn_mean': float(np.mean([x[1] for x in d])), 'scenes': n}
    print(json.dumps(consistency), flush=True)

    runs = [(arm, seed) for arm in CONFIG['models']['arms'] for seed in CONFIG['models']['seeds']]
    if a.limit: runs = [('C0', 42), ('C2', 42)]
    results = {}
    for arm, seed in runs:
        ckpt = E43 / f'checkpoints/{arm}_QB_s{seed}/best.pt'
        net = M43.Model(arm, seed); net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'], strict=True); net = net.to(a.device).eval()
        key = f'{arm}_s{seed}'; results[key] = {'checkpoint_sha256': sha(ckpt)}
        for protocol, path in paths.items():
            rows = {v: [] for v in ('base', 'P5', 'P0')}; diag = {v: [] for v in ('P5', 'P0')}
            with h5py.File(path, 'r') as f:
                for i in range(n):
                    arr = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt') if k in f}
                    x = [torch.from_numpy((arr[k] / 2047).astype('float32')).unsqueeze(0).to(a.device) for k in ('pan', 'lms', 'ms')]
                    with torch.no_grad():
                        y = net(*x).float()
                    outs = {'base': y}
                    for v, margin in MARGINS.items():
                        outs[v], info = proj.project(y.detach(), x[2], margin); diag[v].append(info)
                    for v, out in outs.items():
                        pred = out[0].detach().cpu().numpy().astype(np.float64) * 2047
                        if not np.isfinite(pred).all(): raise FloatingPointError(key)
                        m = rr_metrics(arr['gt'], pred) if protocol == 'RR' else fr_metrics(pred, arr['lms'], arr['ms'], arr['pan'])
                        rows[v].append({'index': i, 'metrics': m})
            mean = lambda rs: {k: float(np.mean([r['metrics'][k] for r in rs])) for k in rs[0]['metrics']}
            archived = json.loads((E43 / f'results/{arm}_QB_s{seed}/evaluation/{protocol}.json').read_text())['mean'] if not a.limit else None
            results[key][protocol] = {
                'mean': {v: mean(r) for v, r in rows.items()},
                'delta_vs_base': {v: {k: mean(rows[v])[k] - mean(rows['base'])[k] for k in rows['base'][0]['metrics']} for v in MARGINS},
                'base_vs_archived_max_abs': None if archived is None else max(abs(mean(rows['base'])[k] - archived[k]) for k in archived),
                'projection': {v: {'iterations_max': max(d['iterations'] for d in ds), 'relative_residual_max': max(d['relative_residual'] for d in ds),
                                   'obs_rmse_before_dn_mean': float(np.mean([d['obs_rmse_before_dn'] for d in ds])), 'obs_rmse_after_dn_mean': float(np.mean([d['obs_rmse_after_dn'] for d in ds]))}
                               for v, ds in diag.items()},
                'rows': rows, 'data_sha256': sha(path)}
            print(key, protocol, 'base', {k: round(v, 5) for k, v in results[key][protocol]['mean']['base'].items()},
                  'dP5', {k: round(v, 5) for k, v in results[key][protocol]['delta_vs_base']['P5'].items()}, flush=True)
        del net
        if a.device == 'cuda': torch.cuda.empty_cache()

    summary, gate = {}, None
    if not a.limit:
        seeds = CONFIG['models']['seeds']
        for arm in CONFIG['models']['arms']:
            summary[arm] = {}
            for protocol in ('RR', 'FR'):
                summary[arm][protocol] = {v: {k: float(np.mean([results[f'{arm}_s{s}'][protocol]['mean'][v][k] for s in seeds])) for k in results[f'{arm}_s{seeds[0]}'][protocol]['mean'][v]}
                                          for v in ('base', 'P5', 'P0')}
        c2, c0 = summary['C2'], summary['C0']
        d_ds = c2['FR']['P5']['D_s'] - c2['FR']['base']['D_s']; d_psnr = c2['RR']['P5']['PSNR'] - c2['RR']['base']['PSNR']
        big_psnr = c2['RR']['P5']['PSNR'] - c0['RR']['base']['PSNR']; big_qnr = c2['FR']['P5']['QNR'] - c0['FR']['base']['QNR']
        gate = {'rule': CONFIG['gate'], 'C2_P5_minus_C2': {'D_s': d_ds, 'PSNR': d_psnr}, 'main_pass': d_ds <= -0.03 and d_psnr >= -0.15,
                'C2_P5_minus_C0': {'PSNR': big_psnr, 'QNR': big_qnr}, 'large_candidate': big_psnr >= 0.3 and big_qnr >= -0.005}
        print('gate', json.dumps(gate), flush=True)
    write(a.output / 'projection.json', {'environment': env, 'consistency': consistency, 'results': results, 'summary_3seed': summary, 'gate': gate})
    write(a.output / 'complete.json', {'gate': gate, 'training_started': False, 'optimizer_updates': 0, 'time': time.time(), 'limit': a.limit})


if __name__ == '__main__':
    main()
