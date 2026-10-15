"""Phase 1: warm start from A0 best (model, Adam state, loader RNG) and train one arm for N epochs."""
import os, sys, json, time, math, argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from common import ssai_suite as suite
from common.model import RestoredPansharpeningNet
from common.metrics import fr_metrics
import importlib
ScreenNet = importlib.import_module('experiments.performance.32_feature_modulation_local_regression_screen.model').ScreenNet


def fr_proxy(m, va, idx):
    m.eval(); rows = []
    with torch.inference_mode():
        for i in idx:
            pan, lms, ms, _ = va[i]
            pred = m(pan[None].cuda(), lms[None].cuda(), ms[None].cuda())[0].cpu().double().numpy() * 2047
            rows.append(fr_metrics(pred, lms.double().numpy() * 2047, ms.double().numpy() * 2047, pan[0].double().numpy() * 2047))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    for k in ('--arm', '--a0', '--cache', '--runtime', '--diag'): ap.add_argument(k, required=True)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--fraction', type=float, default=0.85)
    a = ap.parse_args()
    out = Path(a.runtime) / a.arm; out.mkdir(parents=True, exist_ok=False)
    status = lambda **kw: suite.write(out / 'status.json', {'pid': os.getpid(), 'arm': a.arm, 'updated': time.time(), **kw})
    try:
        eps = json.load(open(a.diag))['D4']['selected_eps']
        suite.configure(42, 'cuda')
        m = ScreenNet(a.arm, eps)
        ck = torch.load(a.a0, map_location='cpu', weights_only=False)
        base_names = list(RestoredPansharpeningNet(4, 4).state_dict())
        missing, unexpected = m.load_state_dict(ck['model'], strict=False)
        new = [n for n, _ in m.named_parameters() if n not in set(base_names)]
        if unexpected or sorted(missing) != sorted(new): raise ValueError(f'warm start mismatch {missing} {unexpected}')
        m.cuda()
        base_params = [p for n, p in m.named_parameters() if n not in set(new)]
        opt = torch.optim.Adam(base_params, lr=1e-4); opt.load_state_dict(ck['optimizer'])
        if new: opt.add_param_group({'params': [p for n, p in m.named_parameters() if n in set(new)]})
        gen = torch.Generator(); gen.set_state(ck['loader_rng'])
        tr = suite.Cached(Path(a.cache) / 'train'); va = suite.Cached(Path(a.cache) / 'val')
        loader = suite.PartialGPUCachedLoader(tr, 32, gen, 'cuda') if a.fraction == 0.85 else None
        vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
        idx = list(range(0, len(va), len(va) // 256))[:256]
        v0, s0 = suite.validate(m, vl, 'cuda')
        if v0 != ck['validation_mse']: raise ValueError(f'start differs from A0: {v0} vs {ck["validation_mse"]}')
        hist = [{'epoch': 0, 'val_mse': v0, 'val_sam': s0}]
        best = v0; best_epoch = 0
        suite.save(out / 'best.pt', {'model': m.state_dict(), 'epoch': 0, 'validation_mse': v0})
        status(state='training', epoch=0, new_parameters=sum(p.numel() for n, p in m.named_parameters() if n in set(new)))
        for ep in range(1, a.epochs + 1):
            for i, total, seen, started in suite.run_epoch(m, opt, loader, 'cuda', micro=32):
                if i % 50 == 0: status(state='training', epoch=ep, batch=i + 1, batches=len(loader), pps=seen / max(time.monotonic() - started, 1e-6), peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            v, s = suite.validate(m, vl, 'cuda')
            if not math.isfinite(v): raise FloatingPointError('nonfinite validation')
            row = {'epoch': ep, 'train_mse': total / seen, 'val_mse': v, 'val_sam': s, 'seconds': time.monotonic() - started}
            if a.arm == 'W2': row['gamma'] = m.gamma.detach().flatten().tolist()
            hist.append(row); suite.write(out / 'history.json', hist); print(json.dumps(row), flush=True)
            if v < best:
                best, best_epoch = v, ep
                suite.save(out / 'best.pt', {'model': m.state_dict(), 'epoch': ep, 'validation_mse': v})
        suite.save(out / 'latest.pt', {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'epoch': a.epochs, 'loader_rng': gen.get_state()})
        status(state='fr_proxy', epoch=a.epochs)
        bk = torch.load(out / 'best.pt', map_location='cpu', weights_only=False); m.load_state_dict(bk['model'])
        proxy = fr_proxy(m, va, idx)
        last3 = float(np.mean([h['val_mse'] for h in hist[-3:]]))
        suite.write(out / 'complete.json', {'arm': a.arm, 'eps': eps, 'start_val_mse': v0, 'best_epoch': best_epoch, 'best_val_mse': best, 'last3_mean_val_mse': last3, 'fr_proxy_best_256': proxy, 'time': time.time()})
        status(state='complete', epoch=a.epochs)
    except Exception as err:
        status(state='failed', reason=repr(err)); raise


if __name__ == '__main__':
    main()
