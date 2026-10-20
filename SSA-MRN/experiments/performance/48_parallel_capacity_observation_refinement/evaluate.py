"""Criteria v2.2 evaluation of 48/49 runs against the reused 43 controls (no training).

Uses common/evaluation_protocol.py (RR, RR2, FR_WALD, FR with observation RMSE) exactly as experiment 47.
The repository `common` package is imported before model.py so that both resolve to the same code.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa: E402
sys.path.insert(0, str(HERE))
import model as M48  # noqa: E402

TOL = {'RR2': (-0.05, 0.06), 'FR_WALD': (-0.07, 0.04), 'FR_OBS': 1.10}  # docs/judgement_criteria.md v2.2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def checkpoint(arm, seed, runtime):
    if arm in ('C0', 'C2'):
        return ROOT / f'experiments/performance/43_normalized_parallel_capacity/checkpoints/{arm}_QB_s{seed}/best.pt'
    return Path(runtime) / f'runs/{arm}_s{seed}/checkpoints/best.pt'


def judge(cand, ctrl):
    """v2.2 on seed means: candidate and control are lists of per-seed mean dicts."""
    mean = lambda rows, part, k: float(np.mean([r[part][k] for r in rows]))
    d = {part: {k: mean(cand, part, k) - mean(ctrl, part, k) for k in ctrl[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
    obs = mean(cand, 'FR', 'OBS_RMSE_DN') / mean(ctrl, 'FR', 'OBS_RMSE_DN')
    rr = {'PSNR': d['RR']['PSNR'] > 0, 'SAM': d['RR']['SAM'] < 0, 'ERGAS': d['RR']['ERGAS'] < 0}
    primary = all(rr.values())
    guards = {k: d[k]['PSNR'] >= TOL[k][0] and d[k]['SAM'] <= TOL[k][1] for k in ('RR2', 'FR_WALD')}
    guards['FR_OBS'] = obs <= TOL['FR_OBS']
    label = '개선' if primary and all(guards.values()) else '부분 개선' if primary or any(rr.values()) else '개선 없음'
    return dict(delta=d, obs_ratio=obs, rr_components_improved=rr, primary=primary, guards=guards, label=label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['C0', 'C2', 'F1', 'WF'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--comparisons', nargs='+', default=['WF:C0', 'WF:C2', 'WF:F1', 'F1:C0'])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    scenes = {}
    for name, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename)); assert len(found) == 1
        with h5py.File(found[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if name == 'RR' else ('pan', 'lms', 'ms')
            scenes[name] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(len(f['pan']))]
        scenes[name + '_sha256'] = sha(found[0])
    degrade = protocol.WaldDegradation('QB', a.device)
    observation = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    models = {}
    for arm in a.arms:
        for seed in a.seeds:
            path = checkpoint(arm, seed, a.runtime)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            net = M48.Model(arm, seed); net.load_state_dict(ck['model'], strict=True); net = net.to(a.device).eval()
            start = time.monotonic()
            rows = [{'index': i, **protocol.scene_metrics(net, rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
            mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
            models[f'{arm}_s{seed}'] = dict(arm=arm, seed=seed, checkpoint=str(path), checkpoint_sha256=sha(path), best_epoch=ck.get('best_epoch'), mean=mean, rows=rows, seconds=time.monotonic() - start)
            print(arm, seed, 'RR %.4f RR2 %.4f FR_WALD %.4f OBS %.3f QNR %.4f' % (mean['RR']['PSNR'], mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], mean['FR']['QNR']), flush=True)
            del net
    judgements = {}
    for pair in a.comparisons:
        cand, ctrl = pair.split(':')
        if all(f'{x}_s{s}' in models for x in (cand, ctrl) for s in a.seeds):
            judgements[pair] = judge([models[f'{cand}_s{s}']['mean'] for s in a.seeds], [models[f'{ctrl}_s{s}']['mean'] for s in a.seeds])
            j = judgements[pair]
            print(pair, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FRW %+.4f/%+.4f OBS x%.3f' % (j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'], j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio']), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(criteria='docs/judgement_criteria.md v2.2', tolerances=TOL, seeds=a.seeds, judgements=judgements, models=models,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=sha(__file__),
                         protocol_sha256=sha(ROOT / 'common/evaluation_protocol.py'), model_sha256=sha(HERE / 'model.py'), rr_sha256=scenes['RR_sha256'], fr_sha256=scenes['FR_sha256'])),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
