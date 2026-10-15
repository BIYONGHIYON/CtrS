"""Fixed-condition training on the verified experiment 4-10 numerical loop."""
import hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ssai_suite as suite
from .model import common_state, new_parameter_count


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FractionGPUCachedLoader(suite.PartialGPUCachedLoader):
    """Experiment 6-10 partial cache with a benchmarked GPU fraction.

    The sample order (inherited __iter__) and the gathered FP32 values are
    identical to the full GPU cache; only the GPU-resident share changes.
    """

    def __init__(self, ds, batch_size, generator, device, fraction):
        self.host = ds.arrays
        self.n = len(ds); self.cut = int(self.n * fraction)
        self.arrays = {k: torch.from_numpy(np.array(v[:self.cut], copy=True)).to(device) for k, v in ds.arrays.items()}
        self.batch_size = batch_size; self.generator = generator
        self.fraction = fraction


def group_changes(initial, current):
    changes = {}
    for i in range(3):
        keys = [k for k in initial if k.startswith(f'ssai_local.{i}.') and k.endswith('.weight')]
        num = sum((current[k].detach().cpu().double() - initial[k].double()).square().sum().item() for k in keys)
        den = sum(initial[k].double().square().sum().item() for k in keys)
        changes[f'stage{i}'] = math.sqrt(num / den)
    return changes


def train(a):
    p = suite.plan(a); out = a.runtime / a.arm; out.mkdir(exist_ok=True)
    with suite.lock(out / 'run.lock'):
        if (out / 'latest.pt').exists(): raise RuntimeError('Automatic restart refused')
        suite.configure(p['seed'], a.device)
        m = suite.model(a.arm).to(a.device)
        if state_hash(common_state(m)) != p['initial_common_sha256']: raise ValueError('Common initialization mismatch')
        initial = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        opt = torch.optim.Adam(m.parameters(), lr=p['lr'])

        def finite_step(optimizer, args, kwargs):
            gradients = [x.grad for x in m.parameters() if x.grad is not None]
            if not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all().item():
                raise FloatingPointError('Nonfinite gradient; update refused')
        opt.register_step_pre_hook(finite_step)
        gen = torch.Generator().manual_seed(p['seed'])
        tr, va = (suite.Cached(Path(p['cache_root']) / split) for split in ('train', 'val'))
        loader = FractionGPUCachedLoader(tr, 32, gen, a.device, p['cache_fraction'])
        vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0, pin_memory=a.device == 'cuda')
        best = float('inf'); history = []; epoch = 0
        suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'starting', 'arm': a.arm, 'new_parameters': new_parameter_count(m), 'cache_fraction': p['cache_fraction'], 'updated': time.time()})
        try:
            for epoch in range(1, p['epochs'] + 1):
                for idx, total, seen, started in suite.run_epoch(m, opt, loader, a.device, micro=p['micro_batch']):
                    if (a.runtime / 'stop.request').exists():
                        suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'stopped', 'epoch': epoch, 'updated': time.time()}); return
                    if idx % 20 == 0:
                        suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'training', 'arm': a.arm, 'epoch': epoch, 'epochs': p['epochs'], 'batch': idx + 1, 'batches': len(loader), 'train_mse': total / seen, 'patches_per_second': seen / max(time.monotonic() - started, 1e-6), 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2, 'updated': time.time()})
                suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'validation', 'arm': a.arm, 'epoch': epoch, 'updated': time.time()})
                vmse, sam = suite.validate(m, vl, a.device)
                if not math.isfinite(vmse) or not math.isfinite(sam): raise FloatingPointError('Nonfinite validation')
                if epoch == 1 and vmse >= p['divergence']['first_epoch_mse_limit']:
                    raise FloatingPointError('Epoch1 validation exceeds fixed 2x A0 epoch1 limit')
                improved = vmse < best; best = min(best, vmse)
                row = {'epoch': epoch, 'train_mse': total / seen, 'val_mse': vmse, 'val_sam': sam, 'val_global_psnr': -10 * math.log10(max(vmse, 1e-15)), 'best_mse': best, 'seconds': time.monotonic() - started}
                row['new_weight_relative_change'] = group_changes(initial, m.state_dict())
                ck = {'config': {'plan': p, 'sensor': 'QB', 'k': 4, 'seed': 42, 'variant': a.arm}, 'model': m.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'best_mse': best, 'validation_mse': vmse, 'loader_rng': gen.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
                if improved: suite.save(out / 'best.pt', ck)
                history.append(row); suite.write(out / 'history.json', history); suite.save(out / 'latest.pt', ck)
                print(json.dumps(row), flush=True)
            suite.write(out / 'complete.json', {'arm': a.arm, 'epochs': p['epochs'], 'best_mse': best, 'best_sha256': suite.sha(out / 'best.pt'), 'latest_sha256': suite.sha(out / 'latest.pt'), 'completed': time.time()})
            suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'complete', 'epoch': p['epochs'], 'updated': time.time()})
        except Exception as error:
            suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'failed', 'reason': str(error), 'epoch': epoch, 'updated': time.time(), 'automatic_restart': False}); raise
