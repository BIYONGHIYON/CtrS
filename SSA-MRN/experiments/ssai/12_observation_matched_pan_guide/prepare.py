"""Experiment 12 preflight without training: verify the reused A0/N1 controls, freeze train-only
guide statistics, check exact N1 equivalence, forwards and zero-update gradients, then write a
fresh plan. No optimizer step is taken and no candidate weight is saved.
"""
import argparse, importlib, json, math, shutil, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import ssai_suite as suite
import h5py, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from common.model import RestoredPansharpeningNet

HERE = Path(__file__).resolve().parent
E8 = ROOT / 'experiments/ssai/8_spatial_mean_normalization'
E4 = ROOT / 'experiments/ssai/4_aligned_sigmoid_gating'
module = importlib.import_module('experiments.ssai.12_observation_matched_pan_guide.model')
n1module = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')
diag = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.diagnostics')
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SNAPSHOT_FILES = ('common/ssai_suite.py', 'common/model.py', 'references/upstream/network.py',
                  'experiments/ssai/8_spatial_mean_normalization/model.py',
                  'experiments/ssai/8_spatial_mean_normalization/train.py',
                  'experiments/ssai/8_spatial_mean_normalization/diagnostics.py')


def build_cache(data_path, split, destination, n, expected):
    """Same /2047 float32 conversion as common/ssai_suite.prepare, accepted only if hashes match."""
    destination.mkdir(parents=True)
    with h5py.File(data_path, 'r') as f:
        for key in suite.KEYS:
            tmp = destination / (key + '.tmp.npy')
            arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=f[key].shape)
            for i in range(0, n, 128):
                arr[i:i + 128] = np.asarray(f[key][i:i + 128], dtype=np.float32) / 2047.
            arr.flush(); del arr
            tmp.replace(destination / (key + '.npy'))
            if suite.sha(destination / (key + '.npy')) != expected[key]:
                raise ValueError(f'Rebuilt cache differs from experiment 8: {split}/{key}')


def train_statistics(train, batch=512):
    """Train-only local (MS-grid 5x5) statistics that fix eps and gain_max."""
    s_p_all, s_b_all = [], []
    for i in range(0, len(train.arrays['pan']), batch):
        pan = torch.from_numpy(np.array(train.arrays['pan'][i:i + batch])).double()
        ms = torch.from_numpy(np.array(train.arrays['ms'][i:i + batch])).double()
        _, s_b, _, s_p = module.observation_statistics(pan, ms, local=True)
        s_p_all.append(s_p.flatten().numpy()); s_b_all.append(s_b.numpy())
    s_p = np.concatenate(s_p_all); s_b = np.concatenate(s_b_all)
    rule = CONFIG['operation']['constants']
    eps = float(np.percentile(s_p, rule['eps_percentile_of_train_local_pan_std']))
    s_p_map = s_p.reshape(s_b.shape[0], 1, *s_b.shape[2:])
    gain = s_b / np.sqrt(s_p_map ** 2 + eps ** 2)
    gain_max = float(np.percentile(gain, rule['gain_max_percentile_of_train_local_gain']))
    q = (0, 1, 5, 50, 95, 99, 99.9, 100)
    return {'eps': eps, 'gain_max': gain_max, 'window_ms_grid': module.WINDOW, 'ratio': module.RATIO,
            'rule': rule, 'source': 'train split only (17,139 patches), float64, before any test input',
            'train_local_pan_std_percentiles': {str(k): float(v) for k, v in zip(q, np.percentile(s_p, q))},
            'train_local_gain_percentiles': {str(k): float(v) for k, v in zip(q, np.percentile(gain, q))},
            'train_gain_clipped_fraction': float((gain > gain_max).mean())}


def guide_summary(pan, ms, stats, local):
    with torch.no_grad():
        _, s_b, _, s_p = module.observation_statistics(pan.double(), ms.double(), local)
        gain = s_b / (s_p * s_p + stats['eps'] ** 2).sqrt()
    return {'gain_median': float(gain.median()), 'gain_clipped_fraction': float((gain > stats['gain_max']).double().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='folder containing train_qb.h5, valid_qb.h5 and QB test H5')
    parser.add_argument('--cache', type=Path, help='existing experiment 4/8 cache; rebuilt and hash-checked when absent')
    a = parser.parse_args()
    if a.runtime.exists(): raise FileExistsError('Fresh runtime required')
    if shutil.disk_usage(a.runtime.parent).free < 20 * 1024 ** 3: raise RuntimeError('At least 20GiB free required')
    old = json.loads((E8 / 'provenance/plan.json').read_text())
    checks = {'passed': False, 'training_started': False, 'optimizer_updates': 0}

    print('phase: environment and source identity with experiment 8 N1', flush=True)
    actual = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}
    for key in ('python', 'torch', 'numpy', 'cuda', 'cudnn', 'gpu'):
        if old['environment'][key] != actual[key]: raise ValueError('Environment mismatch with N1: ' + key)
    with zipfile.ZipFile(E8 / 'provenance/training_source.zip') as z:
        for name in SNAPSHOT_FILES:
            if z.read('source/' + name) != (ROOT / name).read_bytes(): raise ValueError('Differs from N1 training snapshot: ' + name)
    checks['source_identity'] = {'compared_with': 'experiment 8 provenance/training_source.zip', 'files': list(SNAPSHOT_FILES), 'byte_identical': True}

    print('phase: original data and cache', flush=True)
    a.runtime.mkdir(parents=True)
    paths = {}
    for split, filename, n in (('train', 'train_qb.h5', 17139), ('val', 'valid_qb.h5', 1905)):
        found = list(a.data.rglob(filename))
        if len(found) != 1: raise ValueError(found)
        paths[split] = found[0]
        if suite.sha(found[0]) != old['data'][split]['sha256']: raise ValueError('Original data changed: ' + split)
    cache = a.cache
    reuse = cache is not None and all((cache / s / (k + '.npy')).is_file() and suite.sha(cache / s / (k + '.npy')) == h for s in ('train', 'val') for k, h in old['data'][s]['cache'].items())
    if not reuse:
        cache = a.runtime / 'cache'
        for split, n in (('train', 17139), ('val', 1905)):
            build_cache(paths[split], split, cache / split, n, old['data'][split]['cache'])
    checks['cache'] = {'root': str(cache), 'reused_existing': reuse, 'hashes_equal_experiment8': True}

    print('phase: reused control weights', flush=True)
    assets = json.loads((E8 / 'provenance/reference_assets.json').read_text())['files']['A0']
    n1_complete = json.loads((E8 / 'results/training/N1/complete.json').read_text())
    refs = {'A0': {}, 'N1': {}}
    for arm, folder, expected in (('A0', E4 / 'checkpoints/A0_QB_s42', {k: assets[k]['sha256'] for k in ('best', 'latest')}),
                                  ('N1', E8 / 'checkpoints/N1_QB_s42', {k: n1_complete[k + '_sha256'] for k in ('best', 'latest')})):
        for kind in ('best', 'latest'):
            path = folder / (kind + '.pt')
            if suite.sha(path) != expected[kind]: raise ValueError(f'{arm} {kind} hash mismatch')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            net = RestoredPansharpeningNet(4, 4) if arm == 'A0' else n1module.MeanNormalizedNet(4, 4)
            net.load_state_dict(ck['model'], strict=True)
            refs[arm][kind] = {'repository_path': path.relative_to(ROOT).as_posix(), 'sha256': expected[kind], 'epoch': ck['epoch']}
    n1_best = torch.load(E8 / 'checkpoints/N1_QB_s42/best.pt', map_location='cpu', weights_only=False)['model']

    print('phase: freeze train-only guide statistics', flush=True)
    stats = train_statistics(suite.Cached(cache / 'train'))
    checks['observation_statistics'] = stats

    print('phase: initial state, N1 equivalence and forwards', flush=True)
    # Same deterministic/TF32-off settings for the N1 reference and every candidate forward.
    suite.configure(42, 'cuda')
    va = suite.Cached(cache / 'val')
    x = [v.unsqueeze(0).cuda() for v in va[0][:3]]
    reference = n1module.MeanNormalizedNet(4, 4).cuda().eval(); reference.load_state_dict(n1_best, strict=True)
    rectangle = [torch.rand(1, 1, 32, 48, device='cuda'), torch.rand(1, 4, 32, 48, device='cuda'), torch.rand(1, 4, 8, 12, device='cuda')]
    with torch.inference_mode():
        expected_val, expected_rect = reference(*x), reference(*rectangle)
    checks['models'] = {}
    for arm in CONFIG['arms']:
        suite.configure(42, 'cuda')
        net = module.ObservationMatchedNet(4, 4, local=arm == 'B2', eps=stats['eps'], gain_max=stats['gain_max'])
        init_hash = diag.state_hash(net.state_dict())
        if init_hash != old['initial_common_sha256']: raise ValueError('Initial state differs from A0/N1')
        if sum(p.numel() for p in net.parameters()) != sum(p.numel() for p in reference.parameters()): raise ValueError('Parameter count differs from N1')
        net = net.cuda().eval()
        forwards = []
        with torch.inference_mode():
            for h, w in ((64, 64), (256, 256), (512, 512), (32, 48)):
                y = net(torch.rand(1, 1, h, w, device='cuda'), torch.rand(1, 4, h, w, device='cuda'), torch.rand(1, 4, h // 4, w // 4, device='cuda'))
                if y.shape != (1, 4, h, w) or not torch.isfinite(y).all(): raise ValueError('Nonfinite or wrong forward')
                forwards.append([h, w])
            net.load_state_dict(n1_best, strict=True); net.use_matched = False
            if not torch.equal(net(*x), expected_val) or not torch.equal(net(*rectangle), expected_rect): raise ValueError('Raw-PAN path is not exactly N1')
            net.use_matched = True
            changed = float((net(*x) - expected_val).abs().mean())
        checks['models'][arm] = {'initial_common_sha256': init_hash, 'added_parameters': 0, 'finite_forward_shapes': forwards, 'raw_pan_equals_N1_best_exactly': True, 'matched_guide_changes_N1_best_output_mean_abs': changed}
        del net

    print('phase: zero-update diagnostics', flush=True)
    diagnostics = {'optimizer_updates': 0, 'validation_indices': old['diagnostic_validation_indices']}
    batches_val = DataLoader(Subset(va, old['diagnostic_validation_indices']), batch_size=4, shuffle=False, num_workers=0)
    reference_diag = json.loads((E8 / 'results/diagnostics.json').read_text())
    indices = reference_diag['train_gradient_indices']
    batches_train = DataLoader(Subset(suite.Cached(cache / 'train'), indices), batch_size=32, shuffle=False, num_workers=0)
    diagnostics['train_gradient_indices'] = indices
    diagnostics['N1_reference_initial_gradients'] = reference_diag['normalized_initial_gradients']['N1']
    for arm in CONFIG['arms']:
        suite.configure(42, 'cuda')
        net = module.ObservationMatchedNet(4, 4, local=arm == 'B2', eps=stats['eps'], gain_max=stats['gain_max']).cuda()
        diagnostics[arm] = {'initial_validation_rms': diag.rms_ratios(net, batches_val, 'cuda'), 'initial_gradients': diag.gradient_groups(net, batches_train, 'cuda')}
        if diag.state_hash({k: v.cpu() for k, v in net.state_dict().items()}) != old['initial_common_sha256']: raise ValueError('Diagnostics changed weights')
        for stage in diag.STAGES:
            g = diagnostics[arm]['initial_gradients']['groups'][stage]['mean_gradient_l2']
            if not math.isfinite(g) or g == 0: raise ValueError(f'{arm} {stage} gradient not alive')
        del net
    for protocol, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename))
        if len(found) != 1: raise ValueError(found)
        with h5py.File(found[0], 'r') as f:
            pan, ms = (torch.from_numpy(np.asarray(f[k][:], dtype=np.float32) / 2047.) for k in ('pan', 'ms'))
        # Input-only summary recorded after constants were frozen; never used to tune.
        diagnostics[protocol + '_inputs'] = {'scenes': int(pan.shape[0]), 'data_sha256': suite.sha(found[0]), 'B2': guide_summary(pan, ms, stats, True), 'B3': guide_summary(pan, ms, stats, False)}

    print('phase: freeze plan and source', flush=True)
    checks['passed'] = True
    suite.write(a.runtime / 'preflight.json', checks)
    suite.write(a.runtime / 'diagnostics.json', diagnostics)
    p = {k: v for k, v in old.items() if k not in ('reference_runs', 'learning_success', 'followup', 'source_hashes', 'created_utc')}
    p.update(arms=list(CONFIG['arms']), cache_root=str(cache), cache_mode='gpu_partial', cache_fraction=.85, environment=actual,
             reference_runs=refs, observation_statistics=stats, created_utc=time.time(),
             data={s: dict(old['data'][s], path=str(paths[s])) for s in ('train', 'val')}, data_root=str(a.data),
             controls={'A0': 'experiment 4 A0, reused', 'B1': 'experiment 8 N1, reused'},
             automatic={'restart': False, 'condition_change': False, 'evaluation': False, 'seed_repeat': False})
    sources = list((ROOT / 'common').glob('*.py')) + [ROOT / 'references/upstream/network.py'] + [ROOT / n for n in SNAPSHOT_FILES[3:]]
    sources += sorted(HERE.glob('*.py')) + [HERE / 'config.json']
    sources = sorted(set(sources))
    p['source_hashes'] = {path.relative_to(ROOT).as_posix(): suite.sha(path) for path in sources}
    for path in sources:
        dest = a.runtime / 'provenance/source' / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    suite.write(a.runtime / 'plan.json', p); suite.write(a.runtime / 'provenance/plan.json', p)
    with zipfile.ZipFile(a.runtime / 'provenance/training_source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in (a.runtime / 'provenance/source').rglob('*'):
            if path.is_file(): z.write(path, path.relative_to(a.runtime / 'provenance'))
        z.write(a.runtime / 'provenance/plan.json', 'plan.json')
    print(json.dumps({'prepared': True, 'preflight_passed': True, 'training_started': False, 'eps': stats['eps'], 'gain_max': stats['gain_max']}), flush=True)


if __name__ == '__main__': main()
