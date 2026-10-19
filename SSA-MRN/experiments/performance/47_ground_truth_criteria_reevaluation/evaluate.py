"""Experiment 47: evaluate one archived experiment group with the criteria-v2 protocol (no training).

Each group runs in its own process so that experiment-specific module names (normalization33, ...) do not
collide. Models are rebuilt from each experiment's preserved source; the shared network and metric code is the
repository copy, whose equality with the preserved copy (line endings ignored) is recorded.
Tier 1 (groups 29-44) keeps its original loaders; tiers 2/3 (research 4-28, SSAI 1-12) take explicit
checkpoint paths from config.json and rebuild models with the experiment's own model code.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, ast, hashlib, importlib, importlib.util, json, sys, time, types
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/performance'
RESEARCH = ROOT / 'research/performance'
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
    spec = CONFIG['groups'][group]
    if 'paths' in spec:
        template = spec['paths'][arm] if isinstance(spec['paths'], dict) else spec['paths']
        return ROOT / template.format(arm=arm, seed=seed)
    if group == '33_38':
        return EXP / (FOLDERS['33'] if seed == 42 else FOLDERS['38']) / f'checkpoints/{arm}_QB_s{seed}/best.pt'
    if group == '39':
        return EXP / FOLDERS['39'] / f"checkpoints/{'baseline' if arm == 'global' else 'local'}_QB_seed{seed}/best.pt"
    if group in ('41', '42'):
        return EXP / FOLDERS[group] / f'checkpoints/{arm}_QB_seed{seed}/best.pt'
    return EXP / FOLDERS[group] / f'checkpoints/{arm}_QB_s{seed}/best.pt'


def research(folder):
    return importlib.import_module(f'research.performance.{folder}.model')


def embedded_model(run_py):
    """Model source kept as the MODEL string of a notebook driver; the old package import maps to common.model."""
    tree = ast.parse(Path(run_py).read_text(encoding='utf-8'))
    source = next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'MODEL' for t in node.targets))
    source = source.replace('from ssamrn.models.ssa_mrn import RestoredPansharpeningNet', 'from common.model import RestoredPansharpeningNet')
    module = types.ModuleType('embedded_' + Path(run_py).parent.name.split('_')[0]); exec(compile(source, str(run_py), 'exec'), module.__dict__)
    return module


def arm_value(value, arm):
    return value[arm] if isinstance(value, dict) else value


def builder(group):
    """Returns make(arm, seed, ck) -> nn.Module with the experiment's own parameter names."""
    spec = CONFIG['groups'][group]
    loader = spec.get('loader')
    if group == '29':
        M = importlib.import_module('experiments.performance.29_capacity_refinement.model')
        calib = json.loads((EXP / FOLDERS['29'] / 'provenance/plan.json').read_text())['calibration']
        def make(arm, seed, ck):
            net = M.StructureNet(arm, 4); net.configure(calib); return net
        return make
    if group == '30':
        return lambda arm, seed, ck: RestoredPansharpeningNet(4, 4)
    if group in ('33_38', '40'):
        modules = {}
        def make(arm, seed, ck):
            key = 42 if group == '40' else seed
            if key not in modules:
                modules[key] = load_file(f'normalization_{group}_{key}', source_dir(group, seed) / 'normalization.py')
            return modules[key].Model(arm, 0.01)
        return make
    if group in ('39', '41', '42'):
        sys.path.append(str(source_dir(group, 42)))
        name = {'39': 'localnorm_model', '41': 'photoaug_model', '42': 'dilated_model'}[group]
        M = importlib.import_module(name)
        return lambda arm, seed, ck: M.Model(arm)
    if group in ('43', '44'):
        M = load_file(f'exp{group}_model', EXP / FOLDERS[group] / 'model.py')
        return lambda arm, seed, ck: M.Model(arm, seed)
    # Tiers 2/3.
    if loader == 'plain':
        return lambda arm, seed, ck: RestoredPansharpeningNet(4, arm_value(spec['k'], arm))
    if loader == 'r4':
        M = research('4_interpolation_correction_detail')
        return lambda arm, seed, ck: M.make_model(ck['config'])
    if loader == 'build_model':
        M = research(spec['module'])
        return lambda arm, seed, ck: M.build_model(arm, 4, spec['k'])
    if loader == 'r9':
        M = research('9_ssai_fusion')
        return lambda arm, seed, ck: M.FusionPansharpeningNet(4, 6, arm)
    if loader == 'r11':
        M = research('11_haar_reconstruction')
        class Wavelet(M.WaveletMS):
            def forward(self, pan, lms, ms):  # the original model uses PAN and MS only
                return super().forward(pan, ms)
        return lambda arm, seed, ck: Wavelet(arm, 4, ck['config']['width'], ck['config']['blocks'])
    if loader == 'r15':
        M = research('15_shared_reconstruction')
        return lambda arm, seed, ck: M.build(arm)
    if loader == 'variant_class':
        M = research(spec['module'])
        return lambda arm, seed, ck: getattr(M, spec['class'])(arm)
    if loader == 'r23':
        M = research('23_multi_sensor_comparison')
        return lambda arm, seed, ck: M.ComparisonNet(arm, 4)
    if loader == 'r24':
        M = embedded_model(RESEARCH / '24_alpha_bounded_shared_reconstruction/run.py')
        return lambda arm, seed, ck: M.build(arm)
    if loader == 'r28':
        M23, M28 = research('23_multi_sensor_comparison'), research('28_residual_refinement')
        calib = json.loads((RESEARCH / '28_residual_refinement/results/QB/plan.json').read_text())['calibration']
        def make(arm, seed, ck):
            if arm == 'G0': return M23.ComparisonNet('original', 4)
            net = M28.RefinementNet(arm, 4); net.configure(calib); return net
        return make
    if loader == 'ssai':
        suite = importlib.import_module('common.ssai_suite')
        return lambda arm, seed, ck: suite.model(arm)
    if loader == 'ssai8':
        M = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')
        return lambda arm, seed, ck: RestoredPansharpeningNet(4, 4) if arm == 'A0' else M.MeanNormalizedNet(4, 4, learn_gamma=arm == 'N2')
    if loader == 'ssai9':
        M = importlib.import_module('experiments.ssai.9_pan_innovation_correction.model')
        return lambda arm, seed, ck: RestoredPansharpeningNet(4, 4) if arm == 'A0' else M.PanInnovationNet(4, 4, innovation=arm == 'A2')
    if loader == 'ssai12':
        M = importlib.import_module('experiments.ssai.12_observation_matched_pan_guide.model')
        stats = json.loads((ROOT / 'experiments/ssai/12_observation_matched_pan_guide/provenance/plan.json').read_text())['observation_statistics']
        return lambda arm, seed, ck: RestoredPansharpeningNet(4, 4) if arm == 'A0' else M.ObservationMatchedNet(4, 4, local=arm == 'B2', eps=stats['eps'], gain_max=stats['gain_max'])
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
            ck = torch.load(ck_path, map_location='cpu', weights_only=False)
            net = make(arm, seed, ck)
            net.load_state_dict(ck['model'], strict=True)
            net = net.to(a.device).eval()
            start = time.monotonic(); rows = []
            for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR'])):
                rows.append({'index': i, **protocol.scene_metrics(net, rr, fr, a.device, degrade, observation)})
            mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
            models[f'{arm}_s{seed}'] = {'arm': arm, 'seed': seed, 'checkpoint': str(ck_path.relative_to(ROOT)), 'checkpoint_sha256': sha(ck_path),
                                        'checkpoint_epoch': ck.get('epoch'), 'mean': mean, 'rows': rows, 'seconds': time.monotonic() - start}
            print(a.group, arm, seed, 'RR PSNR %.4f RR2 %.4f FR_WALD %.4f OBS %.3f QNR %.4f' % (mean['RR']['PSNR'], mean['RR2']['PSNR'], mean['FR_WALD']['PSNR'], mean['FR']['OBS_RMSE_DN'], mean['FR']['QNR']), flush=True)
            del net, ck
            if a.device == 'cuda': torch.cuda.empty_cache()
    write(out_file, {'group': a.group, 'spec': spec, 'seeds': seeds, 'limit': a.limit, 'code_equality': equality,
                     'environment': {'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
                                     'evaluate_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'common/evaluation_protocol.py'), 'config_sha256': sha(HERE / 'config.json'),
                                     'rr_sha256': scenes['RR_sha256'], 'fr_sha256': scenes['FR_sha256']},
                     'models': models})


if __name__ == '__main__':
    main()
