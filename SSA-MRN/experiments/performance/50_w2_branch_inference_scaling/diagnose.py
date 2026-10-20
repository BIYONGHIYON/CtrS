"""Experiment 50: scale the 43 C2 W2 branch output at inference only (no training) and record RR, RR2, FR_WALD, FR_OBS
with a low/high-band split of the error (low = QB MTF blur of the error, the band the MS observation sees)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa: E402
from common.metrics import rr_metrics  # noqa: E402
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
E43 = ROOT / 'experiments/performance/43_normalized_parallel_capacity'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def load43():
    spec = importlib.util.spec_from_file_location('exp43_model', E43 / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def set_alpha(net, alpha):
    backbone = net.core.backbone
    backbone.forward = lambda x, b=backbone: b.original(x) + alpha * b.branch(x)


def split(pred, gt, blur, device):
    e = torch.from_numpy(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64))[None].to(device)
    with torch.no_grad():
        low = blur(e)
    high = e - low
    return {'RMSE_low_DN': float(low.square().mean().sqrt()), 'RMSE_high_DN': float(high.square().mean().sqrt()), 'RMSE_DN': float(e.square().mean().sqrt())}


def scene(net, rr, fr, device, degrade, observation):
    out = {}
    pred = protocol.predict(net, rr['pan'], rr['lms'], rr['ms'], device)
    out['RR'] = {**rr_metrics(rr['gt'], pred), **split(pred, rr['gt'], observation.blur, device)}
    pan, lms, ms = degrade(rr['pan'], rr['ms'])
    pred = protocol.predict(net, pan, lms, ms, device)
    out['RR2'] = {**rr_metrics(rr['ms'], pred), **split(pred, rr['ms'], observation.blur, device)}
    pan, lms, ms = degrade(fr['pan'], fr['ms'])
    pred = protocol.predict(net, pan, lms, ms, device)
    out['FR_WALD'] = {**rr_metrics(fr['ms'], pred), **split(pred, fr['ms'], observation.blur, device)}
    fused = protocol.predict(net, fr['pan'], fr['lms'], fr['ms'], device)
    out['FR'] = {'OBS_RMSE_DN': protocol.observation_rmse(fused, fr['ms'], observation)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='local smoke test only')
    p.add_argument('--seeds', type=int, nargs='*')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    M43 = load43()
    scenes = {}
    for name, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename)); assert len(found) == 1
        with h5py.File(found[0], 'r') as f:
            n = a.limit or len(f['pan']); keys = ('pan', 'lms', 'ms', 'gt') if name == 'RR' else ('pan', 'lms', 'ms')
            scenes[name] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(n)]
        scenes[name + '_sha256'] = sha(found[0])
    degrade = protocol.WaldDegradation('QB', a.device)
    observation = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    runs = {}
    for seed in a.seeds or CONFIG['seeds']:
        for arm, alphas in (('C0', [None]), ('C2', CONFIG['alphas'])):
            path = E43 / f'checkpoints/{arm}_QB_s{seed}/best.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            for alpha in alphas:
                net = M43.Model(arm, seed); net.load_state_dict(ck['model'], strict=True)
                if alpha is not None: set_alpha(net, alpha)
                net = net.to(a.device).eval(); start = time.monotonic()
                rows = [{'index': i, **scene(net, rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
                mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
                key = f'{arm}_s{seed}' if alpha is None else f'C2_a{alpha:.2f}_s{seed}'
                runs[key] = dict(arm=arm, alpha=alpha, seed=seed, checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path), mean=mean, rows=rows, seconds=time.monotonic() - start)
                print(key, 'RR %.4f RR2 %.4f FRW %.4f OBS %.3f | RR2 low %.3f high %.3f' % (mean['RR']['PSNR'], mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], mean['RR2']['RMSE_low_DN'], mean['RR2']['RMSE_high_DN']), flush=True)
                del net
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config=CONFIG, limit=a.limit, runs=runs, environment=dict(python=sys.version, torch=torch.__version__,
        device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', diagnose_sha256=sha(__file__), config_sha256=sha(HERE / 'config.json'),
        protocol_sha256=sha(ROOT / 'common/evaluation_protocol.py'), model43_sha256=sha(E43 / 'model.py'), rr_sha256=scenes['RR_sha256'], fr_sha256=scenes['FR_sha256'])),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
