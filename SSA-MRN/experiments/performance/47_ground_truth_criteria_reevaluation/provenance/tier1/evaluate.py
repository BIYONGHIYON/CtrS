"""Experiment 47: evaluate one archived experiment group with the criteria-v2 protocol (no training).

Each group runs in its own process so that experiment-specific module names (normalization33, ...) do not
collide. Models are rebuilt from each experiment's preserved source; the shared network and metric code is the
repository copy, whose equality with the preserved copy (line endings ignored) is recorded.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib, importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/performance'
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol
from common.model import RestoredPansharpeningNet
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def code_text(path):
    return Path(path).read_bytes().replace(b'\r\n', b'\n').rstrip(b'\n')


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


FOLDERS = {'29': '29_capacity_refinement', '30': '30_focal_frequency_loss', '33': '33_ms_anchored_reversible_normalization',
           '38': '38_ms_anchored_normalization_seed_repeat', '39': '39_local_ms_anchored_reversible_normalization',
           '40': '40_ms_center_scale_ablation', '41': '41_joint_photometric_augmentation', '42': '42_dilated_backbone_context',
           '43': '43_normalized_parallel_capacity', '44': '44_phase_preserving_frequency_guidance'}


def source_dir(group, seed):
    if group == '33_38':
        return EXP / FOLDERS['33'] / 'provenance/source' if seed == 42 else EXP / FOLDERS['38'] / f'provenance/seed{seed}/source'
    if group in ('39', '41', '42'):
        return EXP / FOLDERS[group] / 'provenance/code'
    if group == '40':
        return EXP / FOLDERS['40'] / 'provenance/source'
    if group in ('43', '44'):
        return EXP / FOLDERS[group] / 'provenance/source'
    return None


def checkpoint(group, arm, seed):
    if group == '33_38':
        return EXP / (FOLDERS['33'] if seed == 42 else FOLDERS['38']) / f'checkpoints/{arm}_QB_s{seed}/best.pt'
    if group == '39':
        return EXP / FOLDERS['39'] / f"checkpoints/{'baseline' if arm == 'global' else 'local'}_QB_seed{seed}/best.pt"
    if group in ('41', '42'):
        return EXP / FOLDERS[group] / f'checkpoints/{arm}_QB_seed{seed}/best.pt'
    return EXP / FOLDERS[group] / f'checkpoints/{arm}_QB_s{seed}/best.pt'


def builder(group):
    if group == '29':
        M = importlib.import_module('experiments.performance.29_capacity_refinement.model')
        calib = json.loads((EXP / FOLDERS['29'] / 'provenance/plan.json').read_text())['calibration']
        def make(arm, seed):
            net = M.StructureNet(arm, 4); net.configure(calib); return net
        return make
    if group == '30':
        return lambda arm, seed: RestoredPansharpeningNet(4, 4)
    if group in ('33_38', '40'):
        modules = {}
        def make(arm, seed):
            key = 42 if group == '40' else seed
            if key not in modules:
                modules[key] = load_file(f'normalization_{group}_{key}', source_dir(group, seed) / 'normalization.py')
            return modules[key].Model(arm, 0.01)
        return make
    if group in ('39', '41', '42'):
        sys.path.append(str(source_dir(group, 42)))
        name = {'39': 'localnorm_model', '41': 'photoaug_model', '42': 'dilated_model'}[group]
        M = importlib.import_module(name)
        return lambda arm, seed: M.Model(arm)
    if group in ('43', '44'):
        M = load_file(f'exp{group}_model', EXP / FOLDERS[group] / 'model.py')
        return lambda arm, seed: M.Model(arm, seed)
    raise ValueError(group)


def load_scenes(data, limit):
    scenes = {}
    for protocol_name, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(data.rglob(filename)); assert len(found) == 1
        with h5py.File(found[0], 'r') as f:
            n = limit or len(f['pan'])
            keys = ('pan', 'lms', 'ms', 'gt') if protocol_name == 'RR' else ('pan', 'lms', 'ms')
            scenes[protocol_name] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(n)]
        scenes[protocol_name + '_sha256'] = sha(found[0])
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=list(CONFIG['groups']))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='local smoke test only')
    parser.add_argument('--seeds', type=int, nargs='*', help='local smoke test only')
    a = parser.parse_args()
    out_file = a.output / f'{a.group}.json'
    if out_file.exists(): raise FileExistsError(out_file)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    spec = CONFIG['groups'][a.group]
    scenes = load_scenes(a.data, a.limit)
    degrade = protocol.WaldDegradation('QB', a.device)
    observation = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    make = builder(a.group)
    seeds = a.seeds or spec['seeds']
    equality = {}
    for seed in seeds:
        src = source_dir(a.group, seed)
        if src is not None:
            equality[str(seed)] = {name: (code_text(src / name) == code_text(ROOT / name)) if (src / name).exists() else None
                                   for name in ('common/model.py', 'references/upstream/network.py')}
    models = {}
    for arm in [spec['control']] + spec['candidates']:
        for seed in seeds:
            ck_path = checkpoint(a.group, arm, seed)
            net = make(arm, seed)
            net.load_state_dict(torch.load(ck_path, map_location='cpu', weights_only=False)['model'], strict=True)
            net = net.to(a.device).eval()
            start = time.monotonic(); rows = []
            for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR'])):
                rows.append({'index': i, **protocol.scene_metrics(net, rr, fr, a.device, degrade, observation)})
            mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
            models[f'{arm}_s{seed}'] = {'arm': arm, 'seed': seed, 'checkpoint': str(ck_path.relative_to(ROOT)), 'checkpoint_sha256': sha(ck_path),
                                        'mean': mean, 'rows': rows, 'seconds': time.monotonic() - start}
            print(a.group, arm, seed, 'RR PSNR %.4f RR2 %.4f FR_WALD %.4f OBS %.3f QNR %.4f' % (mean['RR']['PSNR'], mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], mean['FR']['QNR']), flush=True)
            del net
            if a.device == 'cuda': torch.cuda.empty_cache()
    write(out_file, {'group': a.group, 'spec': spec, 'seeds': seeds, 'limit': a.limit, 'code_equality': equality,
                     'environment': {'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
                                     'evaluate_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'common/evaluation_protocol.py'), 'config_sha256': sha(HERE / 'config.json'),
                                     'rr_sha256': scenes['RR_sha256'], 'fr_sha256': scenes['FR_sha256']},
                     'models': models})


if __name__ == '__main__':
    main()
