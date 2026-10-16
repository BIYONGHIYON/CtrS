"""Experiment 35 phase 0 without training.

G1: cross-scale inconsistency I_cs = MSE(D(f(x_A)), f(x_B)) on validation, rank-correlated with each
archived model's FR test D_s. G2: 8-way dihedral self-ensemble on RR20/FR20 for A0, R0@150, W2@150.
Gate thresholds are fixed in config.json before any measurement.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
from common.interpolation import interp23tap
from common.metrics import rr_metrics, fr_metrics

HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
SB = CONFIG['scale_b']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, path)


def mtf_kernel(gain, ratio=2, size=41):
    """DLPan genMTF construction (same formula as common/observation.mtf_kernels) at the given ratio."""
    coordinate = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(coordinate, coordinate, indexing='ij')
    radial_window = np.interp(np.sqrt(x * x + y * y), np.arange(size // 2 + 1), np.kaiser(size, 0.5)[size // 2:], right=0.0)
    alpha = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(gain)))
    desired = np.exp(-(x * x + y * y) / (2 * alpha * alpha))
    desired[desired < np.finfo(desired.dtype).eps * desired.max()] = 0
    desired /= desired.max()
    impulse = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * radial_window
    return np.real(impulse)


class Degrade2(torch.nn.Module):
    def __init__(self, gains):
        super().__init__()
        self.register_buffer('kernel', torch.from_numpy(np.stack([mtf_kernel(g) for g in gains])[:, None]).float())
        self.phase = tuple(SB['decimation_phase'])

    def forward(self, x):
        pad = self.kernel.shape[-1] // 2
        rows = torch.arange(-pad, x.shape[-2] + pad, device=x.device).clamp(0, x.shape[-2] - 1)
        cols = torch.arange(-pad, x.shape[-1] + pad, device=x.device).clamp(0, x.shape[-1] - 1)
        blurred = F.conv2d(x.index_select(-2, rows).index_select(-1, cols), self.kernel.to(x.dtype), groups=x.shape[1])
        return blurred[..., self.phase[0]::2, self.phase[1]::2]


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def summary_ds(path, arm):
    fr = json.loads(Path(path).read_text())['means'][arm]['FR']
    return fr['D_s'], fr['QNR']


def model_specs():
    """(name, set, constructor, checkpoint, D_s, QNR); FR values from archived evaluations only."""
    s4 = EXP / 'ssai/4_aligned_sigmoid_gating'; s8 = EXP / 'ssai/8_spatial_mean_normalization'; s9 = EXP / 'ssai/9_pan_innovation_correction'
    s5 = EXP / 'ssai/5_mixed_gating'; s6 = EXP / 'ssai/6_ms_conditioned_detail'; s7 = EXP / 'ssai/7_channel_relation_detail'
    s12 = EXP / 'ssai/12_observation_matched_pan_guide'; p29 = EXP / 'performance/29_capacity_refinement'; p34 = EXP / 'performance/34_paper_equation_reproduction'
    M8 = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.model')
    M9 = importlib.import_module('experiments.ssai.9_pan_innovation_correction.model')
    M12 = importlib.import_module('experiments.ssai.12_observation_matched_pan_guide.model')
    M29 = importlib.import_module('experiments.performance.29_capacity_refinement.model')
    M34 = importlib.import_module('experiments.performance.34_paper_equation_reproduction.model')
    calib = json.loads((p29 / 'provenance/plan.json').read_text())['calibration']
    st12 = json.loads((s12 / 'provenance/plan.json').read_text())['observation_statistics']

    def m29(v):
        def make():
            net = M29.StructureNet(v, 4); net.configure(calib); return net
        return make
    specs = [('A0', 'primary', lambda: RestoredPansharpeningNet(4, 4), s4 / 'checkpoints/A0_QB_s42/best.pt', *summary_ds(s4 / 'results/evaluation/summary.json', 'A0'))]
    for arm, g in (('N1', False), ('N2', True)):
        specs.append((f'ssai8_{arm}', 'primary', (lambda g=g: M8.MeanNormalizedNet(4, 4, learn_gamma=g)), s8 / f'checkpoints/{arm}_QB_s42/best.pt', *summary_ds(s8 / 'results/evaluation/summary.json', arm)))
    for arm in ('A1', 'A2'):
        specs.append((f'ssai9_{arm}', 'primary', (lambda arm=arm: M9.PanInnovationNet(4, 4, innovation=arm == 'A2')), s9 / f'checkpoints/{arm}_QB_s42/best.pt', *summary_ds(s9 / 'results/evaluation/summary.json', arm)))
    for target, group in ((50, 'primary'), (150, 'primary'), (100, 'extended')):
        runs = json.loads((p29 / f'results/QB/evaluation/{target}/metrics.json').read_text())['runs']
        for v in ('R0', 'W2', 'W3', 'F01', 'F11'):
            fr = runs[v]['metrics']['FR']['mean']
            specs.append((f'perf29_{v}@{target}', group, m29(v), p29 / f'checkpoints/{v}_QB_s42/best_at_{target}.pt', fr['D_s'], fr['QNR']))
    for arm in ('A1', 'A2', 'A3'):
        specs.append((f'ssai4_{arm}', 'extended', (lambda arm=arm: suite.model(arm)), s4 / f'checkpoints/{arm}_QB_s42/best.pt', *summary_ds(s4 / 'results/evaluation/summary.json', arm)))
    for arm, folder in (('M25', s5), ('M50', s5), ('D6', s6), ('D7', s7)):
        specs.append((f'ssai_{arm}', 'extended', (lambda arm=arm: suite.model(arm)), folder / f'checkpoints/{arm}_QB_s42/best.pt', *summary_ds(folder / 'results/evaluation/summary.json', arm)))
    for arm in ('B2', 'B3'):
        specs.append((f'ssai12_{arm}', 'extended', (lambda arm=arm: M12.ObservationMatchedNet(4, 4, local=arm == 'B2', eps=st12['eps'], gain_max=st12['gain_max'])), s12 / f'checkpoints/{arm}_QB_s42/best.pt', *summary_ds(s12 / 'results/evaluation/summary.json', arm)))
    fr34 = json.loads((p34 / 'results/FR.json').read_text())['mean']
    specs.append(('perf34_PAPER', 'extended', lambda: M34.PaperSSAMRN(4, 6), p34 / 'checkpoints/PAPER_QB_s42/best.pt', fr34['D_s'], fr34['QNR']))
    return specs


def batched(net, tensors, batch=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(tensors[0]), batch):
            out.append(net(*(t[i:i + batch].cuda() for t in tensors)).double().cpu())
    return torch.cat(out)


def spearman(a, b):
    from scipy.stats import spearmanr, pearsonr
    return {'spearman': float(spearmanr(a, b)[0]), 'pearson': float(pearsonr(a, b)[0]), 'n': len(a)}


DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]


def transform(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def inverse(x, k, flip):
    if flip: x = torch.flip(x, (-1,))
    return torch.rot90(x, -k, (-2, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    suite.configure(42, 'cuda')
    env = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'source_sha256': sha(__file__), 'config_sha256': sha(HERE / 'config.json')}

    print('phase: validation data and scale B', flush=True)
    found = list(a.data.rglob('valid_qb.h5'));  assert len(found) == 1
    with h5py.File(found[0], 'r') as f:
        pan, lms, ms, gt = (torch.from_numpy(np.asarray(f[k][:], dtype=np.float32) / 2047.) for k in ('pan', 'lms', 'ms', 'gt'))
    lms_check = float((interp23tap(ms) - lms).abs().max())
    d_ms, d_pan = Degrade2(SB['gnyq_ms_QB']).cuda(), Degrade2([SB['gnyq_pan_QB']]).cuda()
    chunks = lambda op, x: torch.cat([op(x[i:i + 256].cuda().double()).cpu() for i in range(0, len(x), 256)])
    with torch.no_grad():
        pan_b, ms_b, gt_b = chunks(d_pan, pan).float(), chunks(d_ms, ms).float(), chunks(d_ms, gt).float()
        lms_b = interp23tap(ms_b)
    m = SB['interior_margin_scale_b']
    inner = (slice(None), slice(None), slice(m, -m), slice(m, -m))
    data_info = {'validation_sha256': sha(found[0]), 'patches': len(pan), 'h5_lms_vs_interp23tap_max_abs': lms_check,
                 'shapes_b': {'pan': list(pan_b.shape), 'ms': list(ms_b.shape), 'lms': list(lms_b.shape), 'gt': list(gt_b.shape)},
                 'lms_b_vs_gt_b_mse_interior': float(((lms_b.double() - gt_b.double())[inner] ** 2).mean()),
                 'lms_a_vs_gt_a_mse': float(((lms.double() - gt.double()) ** 2).mean())}
    print(json.dumps(data_info), flush=True)

    print('phase: G1 models', flush=True)
    rows = []
    for name, group, make, checkpoint, ds, qnr in model_specs():
        net = make(); net.load_state_dict(load_state(checkpoint), strict=True); net = net.cuda().eval()
        y_a = batched(net, (pan, lms, ms)); y_b = batched(net, (pan_b, lms_b, ms_b))
        with torch.no_grad():
            dy_a = chunks(d_ms, y_a)
        if not (torch.isfinite(y_a).all() and torch.isfinite(y_b).all()): raise ValueError('Nonfinite output: ' + name)
        e_a = float(((y_a - gt.double()) ** 2).mean())
        e_b = float(((y_b - gt_b.double())[inner] ** 2).mean())
        i_cs = float(((dy_a - y_b)[inner] ** 2).mean())
        rows.append({'name': name, 'set': group, 'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha(checkpoint),
                     'I_cs': i_cs, 'E_A': e_a, 'E_B': e_b, 'E_B_over_E_A': e_b / e_a, 'I_cs_over_E_A': i_cs / e_a, 'test_D_s': ds, 'test_QNR': qnr})
        print('%-18s %-8s I_cs=%.4e E_A=%.4e E_B=%.4e Ds=%.4f' % (name, group, i_cs, e_a, e_b, ds), flush=True)
        del net; torch.cuda.empty_cache()
    primary = [r for r in rows if r['set'] == 'primary']
    correlations = {}
    for label, subset in (('primary', primary), ('all', rows)):
        correlations[label] = {metric: {'vs_D_s': spearman([r[metric] for r in subset], [r['test_D_s'] for r in subset]),
                                        'vs_QNR': spearman([r[metric] for r in subset], [r['test_QNR'] for r in subset])}
                               for metric in ('I_cs', 'E_A', 'E_B', 'E_B_over_E_A', 'I_cs_over_E_A')}
    rho_p = correlations['primary']['I_cs']['vs_D_s']['spearman']; rho_all = correlations['all']['I_cs']['vs_D_s']['spearman']
    gate = 'pass' if rho_p >= 0.7 and rho_all >= 0.5 else ('partial' if rho_p >= 0.7 else 'fail')
    write(a.output / 'G1.json', {'environment': env, 'data': data_info, 'models': rows, 'correlations': correlations,
                                 'gate': {'rule': CONFIG['G1']['gate'], 'primary_spearman_I_cs_D_s': rho_p, 'all_spearman_I_cs_D_s': rho_all, 'result': gate}})
    print('G1 gate', gate, rho_p, rho_all, flush=True)

    print('phase: G2 self-ensemble', flush=True)
    specs = {s[0]: s for s in model_specs()}
    g2 = {}
    for name in ('A0', 'perf29_R0@150', 'perf29_W2@150'):
        _, _, make, checkpoint, _, _ = specs[name]
        net = make(); net.load_state_dict(load_state(checkpoint), strict=True); net = net.cuda().eval()
        g2[name] = {}
        for protocol, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
            path = list(a.data.rglob(filename)); assert len(path) == 1
            single, ensemble = [], []
            with h5py.File(path[0], 'r') as f, torch.inference_mode():
                for i in range(len(f['pan'])):
                    keys = ('pan', 'lms', 'ms', 'gt') if protocol == 'RR' else ('pan', 'lms', 'ms')
                    arr = {k: np.asarray(f[k][i], dtype=np.float32) for k in keys}
                    x = [torch.from_numpy(arr[k] / 2047).unsqueeze(0).cuda() for k in ('pan', 'lms', 'ms')]
                    outs = [inverse(net(*(transform(t, k, fl) for t in x)), k, fl)[0].double().cpu() for k, fl in DIHEDRAL]
                    for store, pred in ((single, outs[0]), (ensemble, torch.stack(outs).mean(0))):
                        p = pred.numpy() * 2047
                        store.append(rr_metrics(arr['gt'].astype(np.float64), p) if protocol == 'RR' else fr_metrics(p, arr['lms'], arr['ms'], arr['pan']))
            mean = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            g2[name][protocol] = {'single': mean(single), 'ensemble8': mean(ensemble), 'delta': {k: mean(ensemble)[k] - mean(single)[k] for k in single[0]},
                                  'single_rows': single, 'ensemble_rows': ensemble, 'data_sha256': sha(path[0])}
            print(name, protocol, g2[name][protocol]['delta'], flush=True)
        del net; torch.cuda.empty_cache()
    write(a.output / 'G2.json', {'environment': env, 'method': CONFIG['G2']['method'], 'models': g2})
    write(a.output / 'complete.json', {'G1_gate': gate, 'training_started': False, 'optimizer_updates': 0, 'time': time.time(), 'source_sha256': env['source_sha256']})


if __name__ == '__main__':
    main()
