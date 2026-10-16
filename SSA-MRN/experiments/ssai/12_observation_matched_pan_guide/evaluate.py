"""Experiment 12 full RR20/FR20 evaluation of validation-selected checkpoints; no training.

Arms: A0 and N1 (reused controls, re-evaluated here for the same environment), B2, B3, and
B2Z = B2 best with every SSA block output replaced by zeros (path-use check, not a candidate).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.model import RestoredPansharpeningNet
from common.metrics import rr_metrics, fr_metrics
from common.preview import panels

E4 = ROOT / 'experiments/ssai/4_aligned_sigmoid_gating'
E8 = ROOT / 'experiments/ssai/8_spatial_mean_normalization'
EXAMPLES = (1, 8, 11, 13, 19)
STAGES = ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def select(checkpoint, history_path):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    history = json.loads(Path(history_path).read_text())
    if [h['epoch'] for h in history] != list(range(1, 101)): raise ValueError('Incomplete 100-epoch history')
    minimum = min(h['val_mse'] for h in history)
    earliest = next(h['epoch'] for h in history if h['val_mse'] == minimum)
    if ck['epoch'] != earliest or ck['validation_mse'] != minimum: raise ValueError('best.pt is not the validation-selected epoch')
    return ck, {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint), 'epoch': earliest, 'validation_mse': minimum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    plan = json.loads((a.runtime / 'plan.json').read_text())
    stats = plan['observation_statistics']
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m12 = importlib.import_module('experiments.ssai.12_observation_matched_pan_guide.model')
    n1 = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')

    sources = {
        'A0': (E4 / 'checkpoints/A0_QB_s42/best.pt', E4 / 'results/training/A0/history.json', lambda: RestoredPansharpeningNet(4, 4)),
        'N1': (E8 / 'checkpoints/N1_QB_s42/best.pt', E8 / 'results/training/N1/history.json', lambda: n1.MeanNormalizedNet(4, 4)),
        'B2': (a.runtime / 'B2/best.pt', a.runtime / 'B2/history.json', lambda: m12.ObservationMatchedNet(4, 4, local=True, eps=stats['eps'], gain_max=stats['gain_max'])),
        'B3': (a.runtime / 'B3/best.pt', a.runtime / 'B3/history.json', lambda: m12.ObservationMatchedNet(4, 4, local=False, eps=stats['eps'], gain_max=stats['gain_max'])),
    }
    expected = {'A0': plan['reference_runs']['A0']['best']['sha256'], 'N1': plan['reference_runs']['N1']['best']['sha256']}
    for arm in ('B2', 'B3'):
        expected[arm] = json.loads((a.runtime / arm / 'complete.json').read_text())['best_sha256']
    a.output.mkdir(parents=True)
    selected, checkpoints = {}, {}
    for arm, (cp, hp, _) in sources.items():
        if sha(cp) != expected[arm]: raise ValueError(arm + ' checkpoint hash mismatch')
        checkpoints[arm], selected[arm] = select(cp, hp)
    selected['B2Z'] = {**selected['B2'], 'modification': 'every SSA block output replaced by zeros at evaluation; fusion bias/PReLU kept'}
    write(a.output / 'selection.json', {'frozen_before_evaluation': True, 'rule': 'minimum validation MSE, earliest tie; test not used', 'selected': selected, 'examples_zero_based': list(EXAMPLES), 'precision': 'CUDA FP32', 'no_clipping': True})

    paths = {}
    for protocol, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename))
        if len(found) != 1: raise ValueError(found)
        paths[protocol] = found[0]
    means = {}
    for arm in ('A0', 'N1', 'B2', 'B3', 'B2Z'):
        net = sources['B2' if arm == 'B2Z' else arm][2]()
        net.load_state_dict(checkpoints['B2' if arm == 'B2Z' else arm]['model'], strict=True)
        net = net.cuda().eval()
        hooks = []
        if arm == 'B2Z':
            for stage in STAGES:
                for block in getattr(net, stage):
                    hooks.append(block.register_forward_hook(lambda module, args, output: torch.zeros_like(output)))
        dest = a.output / arm; dest.mkdir(); means[arm] = {}
        for protocol, path in paths.items():
            rows, images, start = [], [], time.monotonic()
            with h5py.File(path, 'r') as f, torch.inference_mode():
                if len(f['pan']) != 20: raise ValueError('Expected 20 scenes')
                for i in range(20):
                    keys = ('pan', 'lms', 'ms', 'gt') if protocol == 'RR' else ('pan', 'lms', 'ms')
                    arrays = {k: np.asarray(f[k][i], dtype=np.float32) for k in keys}
                    x = [torch.from_numpy(arrays[k] / 2047).unsqueeze(0).cuda() for k in ('pan', 'lms', 'ms')]
                    pred = net(*x)[0].cpu().numpy().astype(np.float64) * 2047
                    if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
                    metrics = rr_metrics(arrays['gt'].astype(np.float64), pred) if protocol == 'RR' else fr_metrics(pred, arrays['lms'], arrays['ms'], arrays['pan'])
                    rows.append({'index': i, 'metrics': metrics, 'prediction_min': float(pred.min()), 'prediction_max': float(pred.max())})
                    if protocol == 'RR' and arm in ('B2', 'B3') and i in EXAMPLES:
                        image = dest / f'RR_scene_{i:03}.png'
                        images.append({'index': i, 'path': image.name, **panels(arrays, pred, image, arm, i)})
            mean = {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
            write(dest / (protocol + '.json'), {'arm': arm, 'checkpoint_epoch': selected[arm]['epoch'], 'checkpoint_sha256': selected[arm]['sha256'], 'data_file': path.name, 'data_sha256': sha(path), 'protocol': protocol, 'samples': 20, 'rows': rows, 'mean': mean, 'images': images, 'seconds': time.monotonic() - start, 'precision': 'CUDA FP32 full scene, no clipping', 'limits': 'QNR/Q2n/SCC MATLAB parity unverified; reused exploratory test'})
            means[arm][protocol] = mean
            print(arm, protocol, mean, flush=True)
        for h in hooks: h.remove()
        del net; torch.cuda.empty_cache()
    archived = {'A0': E4 / 'results/evaluation/summary.json', 'N1': E8 / 'results/evaluation/summary.json'}
    consistency = {}
    for arm, path in archived.items():
        if path.exists():
            old = json.loads(path.read_text())['means'][arm]
            consistency[arm] = {p: max(abs(means[arm][p][k] - old[p][k]) for k in old[p]) for p in ('RR', 'FR')}
    write(a.output / 'summary.json', {'means': means, 'archived_mean_max_abs_difference': consistency, 'selection': selected, 'observation_statistics': stats, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'evaluator_sha256': sha(__file__), 'training_started': False})
    write(a.output / 'complete.json', {'full_evaluation_complete': True, 'arms': list(means), 'scene_predictions': 40 * len(means), 'evaluator_sha256': sha(__file__)})


if __name__ == '__main__':
    main()
