"""Validation-selected SSAI 10 evaluation: full RR20/FR20, fresh A0, A0 SSAI-removal diagnostic."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib, json, math, sys, time
from pathlib import Path
import h5py, numpy as np, torch

STAGES = ('SSA_blocks', 'SSA_blocks1', 'SSA_blocks2')
ARMS = ('A0', 'A0Z', 'L2', 'C', 'L1')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); os.replace(tmp, p)


def zero_ssai(net):
    """A0Z: original A0 weights with every original SSA block output replaced by zeros.

    The band fusion convolution, its bias and PReLU still run, so this isolates
    the information carried by the SSA computation itself.
    """
    for stage in STAGES:
        for block in getattr(net, stage):
            block.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True); parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    a = parser.parse_args()
    if 'A0' not in a.arms: parser.error('Fresh A0 evaluation required')
    arms = list(dict.fromkeys(a.arms))
    if a.output.exists(): raise FileExistsError('Use fresh output; earlier reports are never overwritten')
    a.output.mkdir(parents=True); sys.path.insert(0, str(a.source_root))
    from common.model import RestoredPansharpeningNet
    from common.metrics import rr_metrics, fr_metrics
    from common.preview import panels
    module = importlib.import_module('experiments.ssai.10_local_window_cross_attention.model')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((a.runtime / 'plan.json').read_text())

    def model(arm):
        if arm == 'A0': return RestoredPansharpeningNet(4, 4)
        if arm == 'A0Z': return zero_ssai(RestoredPansharpeningNet(4, 4))
        return module.LocalWindowSSAINet(arm)

    selected = {}
    for arm in arms:
        if arm == 'A0Z': continue
        if arm == 'A0':
            cp = Path(plan['baseline_reference']['path']); hp = cp.parent / 'history.json'
            if sha(cp) != plan['baseline_reference']['sha256']: raise ValueError('A0 checkpoint changed')
        else:
            cp = a.runtime / arm / 'best.pt'; hp = a.runtime / arm / 'history.json'
        ck = torch.load(cp, map_location='cpu', weights_only=False); history = json.loads(hp.read_text())
        minimum = min(h['val_mse'] for h in history); earliest = next(h['epoch'] for h in history if h['val_mse'] == minimum)
        assert ck['epoch'] == earliest and ck['validation_mse'] == minimum
        assert len(history) == 100 and [h['epoch'] for h in history] == list(range(1, 101))
        selected[arm] = {'checkpoint': str(cp), 'sha256': sha(cp), 'epoch': ck['epoch'], 'validation_mse': minimum}
    if 'A0Z' in arms: selected['A0Z'] = {**selected['A0'], 'same_checkpoint_as': 'A0', 'diagnostic': 'original SSA block outputs zeroed; not a candidate'}
    write(a.output / 'selection.json', {'frozen_before_evaluation': True, 'rule': 'minimum validation MSE, earliest tie; test not used', 'selected': selected, 'examples_zero_based': [1, 8, 11, 13, 19], 'precision': 'FP32', 'no_clipping': True})
    allrows = {}; profiles = {}
    for arm in arms:
        cp = Path(selected[arm]['checkpoint']); ck = torch.load(cp, map_location='cpu', weights_only=False)
        net = model(arm); net.load_state_dict(ck['model'], strict=True); net.to(a.device).eval(); dest = a.output / arm; dest.mkdir(); record = {}
        for protocol, filename in [('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')]:
            paths = list(a.data.rglob(filename)); assert len(paths) == 1; path = paths[0]; rows = []; images = []; start = time.monotonic()
            with h5py.File(path, 'r') as f, torch.inference_mode():
                assert len(f['pan']) == 20
                for i in range(len(f['pan'])):
                    keys = ('pan', 'lms', 'ms', 'gt') if protocol == 'RR' else ('pan', 'lms', 'ms')
                    arrays = {k: np.asarray(f[k][i], dtype=np.float32) for k in keys}
                    x = [torch.from_numpy(arrays[k] / 2047).unsqueeze(0).to(a.device) for k in ('pan', 'lms', 'ms')]
                    pred = net(*x)[0].cpu().numpy().astype(np.float64) * 2047
                    assert np.isfinite(pred).all()
                    metrics = rr_metrics(arrays['gt'].astype(np.float64), pred) if protocol == 'RR' else fr_metrics(pred, arrays['lms'], arrays['ms'], arrays['pan'])
                    assert all(np.isfinite(v) for v in metrics.values())
                    rows.append({'index': i, 'metrics': metrics, 'prediction_min': float(pred.min()), 'prediction_max': float(pred.max())})
                    if protocol == 'RR' and i in (1, 8, 11, 13, 19) and arm != 'A0Z':
                        image = dest / f'scene_{i:02}.png'; images.append({'index': i, 'path': image.name, **panels(arrays, pred, image, arm, i)})
            result = {'arm': arm, 'checkpoint_epoch': ck['epoch'], 'checkpoint_sha256': sha(cp), 'data_file': filename, 'data_sha256': sha(path), 'protocol': protocol, 'samples': len(rows), 'rows': rows, 'mean': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}, 'images': images, 'seconds': time.monotonic() - start, 'precision': a.device + ' FP32 full scene, no clipping', 'limits': 'QNR/Q2n/SCC MATLAB parity unverified; reused exploratory test, not independent final holdout'}
            write(dest / (protocol + '.json'), result); record[protocol] = result; print(arm, protocol, result['mean'], flush=True)
        allrows[arm] = record

        def synchronize():
            if a.device == 'cuda': torch.cuda.synchronize()
        profile = {'registered_parameters': sum(p.numel() for p in net.parameters()), 'new_parameters': sum(p.numel() for n, p in net.named_parameters() if n.startswith('ssai_local.')), 'timing': a.device + ' isolated full forward; synthetic inputs; 3 warmups / 10 repetitions', 'shapes': []}
        for size in (64, 256, 512):
            torch.manual_seed(42); x = [torch.rand(1, 1, size, size, device=a.device), torch.rand(1, 4, size, size, device=a.device), torch.rand(1, 4, size // 4, size // 4, device=a.device)]; macs = [0]

            def count(m, inputs, output): macs[0] += output.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
            hooks = [m.register_forward_hook(count) for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
            with torch.inference_mode(): net(*x)
            for h in hooks: h.remove()
            # q.k and weighted value sum: 2 * width * window^2 MACs per pixel and band.
            attention = sum(4 * (size // 2 ** s) ** 2 * 2 * module.WIDTH * module.WINDOW ** 2 for s in range(3)) if arm in ('L1', 'L2') else 0
            times = []
            with torch.inference_mode():
                for _ in range(3): net(*x)
                synchronize()
                if a.device == 'cuda': torch.cuda.reset_peak_memory_stats()
                for _ in range(10):
                    synchronize(); tick = time.perf_counter(); net(*x); synchronize(); times.append((time.perf_counter() - tick) * 1000)
            profile['shapes'].append({'size': size, 'executed_conv_macs': macs[0], 'attention_macs': attention, 'latency_median_ms': float(np.median(times)), 'latency_p95_ms': float(np.percentile(times, 95)), 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2 if a.device == 'cuda' else 0, 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2 if a.device == 'cuda' else 0})
        profiles[arm] = profile; write(dest / 'profile.json', profile); del net, ck
        if a.device == 'cuda': torch.cuda.empty_cache()
    differences = {}
    for arm in [x for x in arms if x != 'A0']:
        differences[arm] = {}
        for protocol in ('RR', 'FR'):
            base = allrows['A0'][protocol]; candidate = allrows[arm][protocol]
            differences[arm][protocol] = {'mean_delta': {k: candidate['mean'][k] - base['mean'][k] for k in base['mean']}, 'scenes': [{'index': b['index'], 'delta': {k: c['metrics'][k] - b['metrics'][k] for k in b['metrics']}} for b, c in zip(base['rows'], candidate['rows'])]}
    summary = {'means': {arm: {p: d['mean'] for p, d in rows.items()} for arm, rows in allrows.items()}, 'differences': differences, 'profiles': profiles, 'selection': selected, 'scene_counts': {'RR': 20, 'FR': 20}, 'total_scene_predictions': 40 * len(arms), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else None, 'evaluator_sha256': sha(__file__)}
    write(a.output / 'summary.json', summary)
    write(a.output / 'complete.json', {'full_evaluation_complete': True, 'scene_predictions': 40 * len(arms), 'evaluator_sha256': sha(__file__), 'time': time.time()})


if __name__ == '__main__': main()
