"""Verify A0 assets, run SSAI 10 preflight (D1 removal, D2 size invariance, gradients), freeze plan."""
import argparse, importlib, json, math, shutil, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
module = importlib.import_module('experiments.ssai.10_local_window_cross_attention.model')
helpers = importlib.import_module('experiments.ssai.10_local_window_cross_attention.train')
ARMS = ['L2', 'C', 'L1']  # queue order: L2 and C first, L1 when a slot frees
A0_INITIAL = 'c24fee837e68d93a796b47949a4b7dfde0cf1860734532ea2b39cbf2ded56ea6'


def stage_inputs(size, generator):
    g = generator
    return (torch.rand(1, 1, size, size, generator=g, dtype=torch.float64), torch.rand(1, 4, size, size, generator=g, dtype=torch.float64),
            torch.rand(1, 4, size, size, generator=g, dtype=torch.float64), torch.randn(1, 64, size, size, generator=g, dtype=torch.float64))


def invariance(fn, margin=8, size=96, crop=64, offset=16):
    """Max |full interior - crop interior| for a local operator; 0 means size-independent interior."""
    x = stage_inputs(size, torch.Generator().manual_seed(7))
    full = fn(*x)
    part = fn(*[t[..., offset:offset + crop, offset:offset + crop] for t in x])
    a = full[..., offset + margin:offset + crop - margin, offset + margin:offset + crop - margin]
    b = part[..., margin:crop - margin, margin:crop - margin]
    return float((a - b).abs().max()), float(a.abs().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True, help='experiment 4 runtime holding the original A0')
    a = parser.parse_args()
    if a.runtime.exists(): raise FileExistsError('Fresh runtime required')
    if shutil.disk_usage(a.previous).free < 20 * 1024 ** 3: raise RuntimeError('Need 20 GiB free')
    old = json.loads((a.previous / 'plan.json').read_text())
    actual = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}
    for k, v in actual.items():
        if old['environment'][k] != v: raise ValueError('Environment changed: ' + k)
    for name in ('common/model.py', 'references/upstream/network.py'):
        if suite.sha(ROOT / name) != old['source_hashes'][name]: raise ValueError('Original architecture changed')
    cache = Path(old.get('cache_root', a.previous / 'cache'))
    print('phase: data, cache and A0 checkpoint hashes', flush=True)
    for split in ('train', 'val'):
        info = old['data'][split]
        if suite.sha(info['path']) != info['sha256']: raise ValueError('Original data changed')
        for key, h in info['cache'].items():
            if suite.sha(cache / split / (key + '.npy')) != h: raise ValueError('Cache changed')
    cp = a.previous / 'A0/best.pt'; cp_hash = suite.sha(cp)
    if cp_hash != json.loads((a.previous / 'A0/complete.json').read_text())['best_sha256']: raise ValueError('A0 weight changed')
    state = torch.load(cp, map_location='cpu', weights_only=False)
    a0_history = json.loads((a.previous / 'A0/history.json').read_text())

    print('phase: explicit attention backward against plain autograd reference (float64)', flush=True)
    g = torch.Generator().manual_seed(11)
    tensors = [torch.randn(*shape, generator=g, dtype=torch.float64) for shape in ((2, 4, 2, 8, 12, 10), (2, 2, 8, 12, 10), (2, 2, 8, 12, 10), (4, 2, 49))]
    results = {}
    for name, fn in (('explicit', module.local_window_attention), ('reference', module.reference_local_window_attention)):
        inputs = [t.clone().cuda().requires_grad_() for t in tensors]
        out = fn(*inputs)
        weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(12), dtype=torch.float64).cuda()
        (out * weight).sum().backward()
        results[name] = [out.detach()] + [t.grad for t in inputs]
    attention_equivalence = [float((x - y).abs().max()) for x, y in zip(results['explicit'], results['reference'])]
    if max(attention_equivalence) > 1e-10: raise ValueError(f'Explicit backward differs from reference: {attention_equivalence}')

    print('phase: common initialization, parameters, gradients, shapes', flush=True)
    suite.configure(42, 'cuda'); control = RestoredPansharpeningNet(4, 4)
    initial = {k: v.clone() for k, v in control.state_dict().items()}
    init_hash = helpers.state_hash(initial)
    if init_hash != A0_INITIAL: raise ValueError('Not experiment 4 A0 common initial state')
    va = suite.Cached(cache / 'val')
    batch = [torch.stack([va[i][j] for i in range(2)]).cuda() for j in range(4)]
    checks = {'passed': False, 'attention_explicit_vs_reference_max_abs_diff': {'output_q_k_v_bias': attention_equivalence, 'dtype': 'float64', 'device': 'cuda'}, 'initial_common_sha256': init_hash, 'a0_registered_parameters': sum(p.numel() for p in control.parameters()), 'models': {}}
    for arm in ARMS:
        suite.configure(42, 'cuda'); net = module.LocalWindowSSAINet(arm)
        if helpers.state_hash(module.common_state(net)) != init_hash: raise ValueError('Common initial state differs: ' + arm)
        net.cuda().train()
        F.mse_loss(net(*batch[:3]), batch[3]).backward()  # deterministic-mode backward on CUDA
        grads = {n: float(p.grad.norm()) for n, p in net.named_parameters() if n.startswith('ssai_local.')}
        missing = [n for n, p in net.named_parameters() if n.startswith('ssai_local.') and p.grad is None]
        if missing or not all(math.isfinite(v) and v > 0 for v in grads.values()): raise ValueError(f'{arm} gradient check failed: {missing}')
        net.zero_grad(set_to_none=True); net.eval(); shapes = []
        with torch.inference_mode():
            for h, w in ((64, 64), (256, 256), (512, 512), (32, 48)):
                x = [torch.rand(1, 1, h, w, device='cuda'), torch.rand(1, 4, h, w, device='cuda'), torch.rand(1, 4, h // 4, w // 4, device='cuda')]
                y = net(*x)
                if y.shape != (1, 4, h, w) or not torch.isfinite(y).all(): raise ValueError('Shape/finite check failed')
                shapes.append([h, w])
        stage = net.ssai_local[0].double().cpu()
        with torch.no_grad():
            diff, scale = invariance(lambda pan, feat, ms, bb: stage(pan, feat, ms, bb))
        if diff > 1e-10: raise ValueError(f'{arm} local operator depends on image size: {diff}')
        checks['models'][arm] = {'registered_parameters': sum(p.numel() for p in net.parameters()), 'new_parameters': module.new_parameter_count(net), 'new_parameter_gradient_norms': grads, 'shapes': shapes, 'common_initial_match': True, 'size_invariance_max_abs_diff': diff, 'size_invariance_output_mean_abs': scale}
        del net, stage; torch.cuda.empty_cache()
    block = RestoredPansharpeningNet(4, 4)
    block.load_state_dict(state['model'], strict=True)
    original = block.SSA_blocks[0].double()
    with torch.no_grad():
        diff, scale = invariance(lambda pan, feat, ms, bb: original(pan, torch.cat([feat, ms[:, :1]], 1)))
    checks['original_ssa_block_size_invariance'] = {'max_abs_diff': diff, 'output_mean_abs': scale, 'note': 'global spatial softmax over the whole input; trained A0 band-0 full-resolution block'}

    print('phase: D1 A0 validation with original SSA outputs removed', flush=True)
    control.load_state_dict(state['model'], strict=True); control.cuda().eval()
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0)
    normal = suite.validate(control, vl, 'cuda')
    hooks = [blk.register_forward_hook(lambda m, i, o: torch.zeros_like(o)) for s in module.STAGES for blk in getattr(control, s)]
    removed = suite.validate(control, vl, 'cuda')
    for h in hooks: h.remove()
    recorded = state['validation_mse']
    diagnostics = {'D1_ssai_removal': {'validation_samples': len(va), 'a0_best_epoch': state['epoch'], 'a0_recorded_validation_mse': recorded, 'a0_validation_mse': normal[0], 'a0_validation_sam': normal[1], 'removed_validation_mse': removed[0], 'removed_validation_sam': removed[1], 'relative_mse_change_percent': 100 * (removed[0] - normal[0]) / normal[0], 'definition': 'every original SSA block output replaced by zeros; fusion conv bias and PReLU kept'},
                   'D2_size_invariance': {arm: checks['models'][arm]['size_invariance_max_abs_diff'] for arm in ARMS} | {'original_A0_block': checks['original_ssa_block_size_invariance']['max_abs_diff']}}
    checks['a0_validation_reproduced'] = normal[0] == recorded
    a.runtime.mkdir(parents=True)
    suite.write(a.runtime / 'diagnostics.json', diagnostics)
    checks['passed'] = True; suite.write(a.runtime / 'preflight.json', checks)
    first = next(r['val_mse'] for r in a0_history if r['epoch'] == 1)
    p = {k: v for k, v in old.items() if k not in ('initial_checkpoints', 'optimization_benchmark', 'vram_safety', 'reference_area_candidate', 'reference_runs')}
    p.update(arms=ARMS, experiment=10, research_track='ssai', cache_root=str(cache), cache_mode='gpu_partial', cache_fraction=0.85, micro_batch=32, effective_batch=32, initial_common_sha256=init_hash, environment=actual, created_utc=time.time(), baseline_reference={'path': str(cp), 'sha256': cp_hash, 'epoch': state['epoch']}, divergence={'first_epoch_mse_limit': 2 * first, 'reference': '2*A0 epoch1 validation', 'automatic_restart': False}, followup={'automatic_launch': False, 'queue': 'L2 and C in parallel, L1 when a slot frees, then evaluation'}, source_hashes={})
    paths = list((ROOT / 'common').glob('*.py')) + [ROOT / 'references/upstream/network.py'] + list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'config.json']
    for path in paths:
        name = path.relative_to(ROOT); p['source_hashes'][name.as_posix()] = suite.sha(path)
        dest = a.runtime / 'provenance/source' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    suite.write(a.runtime / 'plan.json', p); suite.write(a.runtime / 'provenance/plan.json', p)
    suite.save(a.runtime / 'provenance/initial_state.pt', initial)
    with zipfile.ZipFile(a.runtime / 'provenance/training_source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in (a.runtime / 'provenance').rglob('*'):
            if path.is_file() and path.suffix != '.zip': z.write(path, path.relative_to(a.runtime / 'provenance'))
    print(json.dumps({'preflight': {k: v for k, v in checks.items() if k != 'models'}, 'models': {k: {x: y for x, y in v.items() if x != 'new_parameter_gradient_norms'} for k, v in checks['models'].items()}, 'diagnostics': diagnostics}), flush=True)


if __name__ == '__main__': main()
