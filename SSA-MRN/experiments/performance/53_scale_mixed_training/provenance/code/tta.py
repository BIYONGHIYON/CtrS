"""Training-free check: x8 geometric self-ensemble (rotations x flips, averaged after inverting) for F1 and C2.

Same criteria-v2.3 protocol as experiment 48/49 (common/evaluation_protocol.scene_metrics). Note: flipping both PAN
and MS changes the MS sampling phase relative to PAN by one PAN pixel (dataset phase 2 of 4), so the ensemble
members see a slightly different PAN/MS alignment than in training. Reported as a measured fact, not corrected.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa: E402
E48 = HERE.parent / '48_parallel_capacity_observation_refinement'
sys.path.insert(0, str(E48))
import evaluate as EV48  # noqa: E402  (judge, checkpoint paths, model module)


class SelfEnsemble(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, pan, lms, ms):
        outs = []
        for flip in (False, True):
            for k in range(4):
                t = lambda x: torch.rot90(torch.flip(x, (-1,)) if flip else x, k, (-2, -1))
                y = self.net(t(pan), t(lms), t(ms))
                y = torch.rot90(y, -k, (-2, -1))
                outs.append(torch.flip(y, (-1,)) if flip else y)
        return torch.stack(outs).mean(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
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
    degrade = protocol.WaldDegradation('QB', a.device)
    observation = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    models = {}
    for arm in ('C0', 'F1', 'C2'):
        for seed in a.seeds:
            path = (ROOT / f'experiments/performance/49_observation_refinement_seed_repeat/checkpoints/F1_QB_s{seed}/best.pt' if arm == 'F1'
                    else ROOT / f'experiments/performance/43_normalized_parallel_capacity/checkpoints/{arm}_QB_s{seed}/best.pt')
            net = EV48.M48.Model(arm, seed); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
            for tag, model in (('plain', net), ('tta8', SelfEnsemble(net))):
                model = model.to(a.device).eval(); start = time.monotonic()
                rows = [{'index': i, **protocol.scene_metrics(model, rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
                mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
                models[f'{arm}_{tag}_s{seed}'] = dict(arm=arm, tta=tag, seed=seed, checkpoint_sha256=sha(path), mean=mean, rows=rows, seconds=time.monotonic() - start)
                print(arm, tag, seed, 'RR %.4f RR2 %.4f FRW %.4f OBS %.3f' % (mean['RR']['PSNR'], mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN']), flush=True)
    judgements = {}
    for arm in ('C0', 'F1', 'C2'):
        judgements[f'{arm}_tta8_vs_{arm}_plain'] = EV48.judge([models[f'{arm}_tta8_s{s}']['mean'] for s in a.seeds], [models[f'{arm}_plain_s{s}']['mean'] for s in a.seeds])
    judgements['C2_tta8_vs_F1_plain'] = EV48.judge([models[f'C2_tta8_s{s}']['mean'] for s in a.seeds], [models[f'F1_plain_s{s}']['mean'] for s in a.seeds])
    for k, j in judgements.items():
        print(k, j['label'], 'RR %+.4f RR2 %+.4f FRW %+.4f OBS x%.3f' % (j['delta']['RR']['PSNR'], j['delta']['RR2']['PSNR'], j['delta']['FR_WALD']['PSNR'], j['obs_ratio']), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(seeds=a.seeds, judgements=judgements, models=models, environment=dict(torch=torch.__version__, tta_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'common/evaluation_protocol.py'))),
                                   ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
