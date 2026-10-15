"""Experiment8 training, reusing the verified numerical training and validation path."""
import hashlib, importlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from common import ssai_suite as suite
from .diagnostics import preserve_rng, rms_ratios, state_hash, weight_changes, STAGES


def train(a):
    p = suite.plan(a)
    out = a.runtime / a.arm
    out.mkdir(exist_ok=True)
    with suite.lock(out / 'run.lock'):
        if (out / 'latest.pt').exists():
            raise RuntimeError('No automatic restart: explicit resume authorization required')
        suite.configure(p['seed'], a.device)
        m = suite.model(a.arm).to(a.device)
        initial = {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if not k.endswith('.gamma')}
        if state_hash(initial) != p['initial_common_sha256']:
            raise ValueError('Common initialization differs from verified A0')
        opt = torch.optim.Adam(m.parameters(), lr=p['lr'])
        # Reject nonfinite gradients before Adam mutates any parameter/state.
        def finite_step(optimizer, args, kwargs):
            norms = torch._foreach_norm([x.grad for x in m.parameters() if x.grad is not None])
            if not torch.isfinite(torch.stack(norms)).all().item():
                raise FloatingPointError('Nonfinite gradient; Adam step refused')
        opt.register_step_pre_hook(finite_step)
        gen = torch.Generator().manual_seed(p['seed'])
        fingerprint = hashlib.sha256(json.dumps({'plan': p, 'arm': a.arm}, sort_keys=True).encode()).hexdigest()
        cache = Path(p['cache_root'])
        tr, va = suite.Cached(cache / 'train'), suite.Cached(cache / 'val')
        loader = suite.make_loader(tr, p['cache_mode'], 32, gen, a.device)
        vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0, pin_memory=a.device == 'cuda')
        diagnostic_loader = DataLoader(Subset(va, p['diagnostic_validation_indices']), batch_size=4, shuffle=False, num_workers=0)
        best = float('inf'); history = []
        try:
            for epoch in range(1, p['epochs'] + 1):
                for idx, total, seen, started in suite.run_epoch(m, opt, loader, a.device, micro=32):
                    if (a.runtime / 'stop.request').exists():
                        suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'stopped', 'epoch': epoch, 'updated': time.time()})
                        return
                    if idx % 20 == 0:
                        suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'training', 'arm': a.arm, 'epoch': epoch, 'epochs': p['epochs'], 'batch': idx+1, 'batches': len(loader), 'train_mse': total/seen, 'patches_per_second': seen/max(time.monotonic()-started, 1e-6), 'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2, 'updated': time.time()})
                suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'validation', 'epoch': epoch, 'updated': time.time()})
                vmse, sam = suite.validate(m, vl, a.device)
                if not math.isfinite(vmse) or not math.isfinite(sam):
                    raise FloatingPointError('Nonfinite validation')
                if epoch == 1 and vmse >= p['divergence']['first_epoch_mse_limit']:
                    suite.write(out / 'divergence.json', {'epoch': epoch, 'val_mse': vmse, 'limit': p['divergence']['first_epoch_mse_limit'], 'reference_epoch': 1, 'automatic_restart': False})
                    raise FloatingPointError('Epoch1 validation MSE exceeds the fixed A0 epoch1 limit')
                improved = vmse < best
                best = min(best, vmse)
                row = {'epoch': epoch, 'train_mse': total/seen, 'val_mse': vmse, 'val_sam': sam, 'val_global_psnr': -10*math.log10(max(vmse, 1e-15)), 'best_mse': best, 'seconds': time.monotonic()-started}
                with preserve_rng():
                    row['weight_relative_change'] = weight_changes(m, initial)
                    row['gamma'] = {f'{stage}.{i}': float(block.gamma.detach().cpu()) for stage in STAGES for i, block in enumerate(getattr(m, stage)) if isinstance(block.gamma, torch.Tensor)}
                    if epoch in (1, 10, 50, 100):
                        row['rms_ratios'] = rms_ratios(m, diagnostic_loader, a.device)
                state = {'config': {'plan': p, 'sensor': 'QB', 'k': 4, 'seed': 42, 'variant': a.arm}, 'fingerprint': fingerprint, 'model': m.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'best_mse': best, 'validation_mse': vmse, 'loader_rng': gen.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
                if improved: suite.save(out / 'best.pt', state)
                history.append(row)
                suite.write(out / 'history.json', history)
                suite.save(out / 'latest.pt', state)
                print(json.dumps(row), flush=True)
            suite.write(out / 'complete.json', {'arm': a.arm, 'epochs': 100, 'best_mse': best, 'best_sha256': suite.sha(out/'best.pt'), 'latest_sha256': suite.sha(out/'latest.pt'), 'fingerprint': fingerprint, 'completed': time.time()})
            suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'complete', 'epoch': 100, 'updated': time.time()})
        except Exception as error:
            suite.write(out / 'status.json', {'pid': os.getpid(), 'state': 'failed', 'reason': str(error), 'updated': time.time(), 'automatic_restart': False})
            raise
