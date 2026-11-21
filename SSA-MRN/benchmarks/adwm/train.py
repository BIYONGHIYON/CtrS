"""ADWM retrain on our PanCollection split, following the official main_train.py with the paper settings.

Official loop kept as is: seed -> BWNet -> DataLoader(batch 64, shuffle, drop_last, train arrays gt / pan / ms in DN / scale)
-> L1 on model(ms, pan) -> Adam(lr 2e-3, weight decay 0, 'initial_lr') -> StepLR(step 150, gamma 0.5, last_epoch=0) stepped
once per epoch. That construction already uses the halved rate in epoch index 149, 299, 449 (0-based; the 150th, 300th,
450th epoch), one epoch earlier than a default StepLR (index 150, 300, 450).
Epochs: QB 200, GF2 500 (paper supplement D). QB uses the IFW hidden size 0.6n (model.py). cudnn deterministic as official.

Team rules added (AGENTS.md section 8): seed 42; full validation L1 after every epoch (official: every 10 epochs, shuffled,
drop_last, no selection) and best.pt = minimum validation L1; latest.pt carries everything needed to resume exactly
(model, optimizer, scheduler, RNG states, history), and --deadline stops cleanly at an epoch boundary for Kaggle sessions.

--recompute: layer recomputation (torch.utils.checkpoint) of the six LAC residual blocks, only when batch 64 does not fit in
GPU memory. BatchNorm running statistics would be updated twice by the recomputed forward, so they are saved before and
restored after every recomputed BatchNorm call; the result is then the same as without recomputation (checked on CPU).

python train.py train --sensor QB --data /kaggle/input --out /kaggle/working/adwm/QB --device cuda:0 --deadline-hours 5.5
python train.py train --sensor QB --data /kaggle/input --out /kaggle/working/adwm/QB --resume /kaggle/input/.../latest.pt
python train.py measure --sensor GF2 --device cuda:1            # GPU memory and seconds per iteration, batch 64
python train.py author-select --data /kaggle/input --device cuda  # WV3 validation L1 of the two LayerNorm handlings
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('adwm_adapter', HERE / 'model.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

SENSORS = {'QB': ('qb', 2047., 4), 'GF2': ('gf2', 1023., 4), 'WV3': ('wv3', 2047., 8)}
EPOCHS = {'QB': 200, 'GF2': 500}        # paper supplement D
TRAIN_SIZE = {'QB': 17139, 'GF2': 19809}
VALID_SIZE = {'QB': 1905, 'GF2': 2201}
BATCH, LR, STEP, GAMMA, SEED = 64, 2e-3, 150, 0.5, 42
KST = timezone(timedelta(hours=9))


def now(): return datetime.now(KST).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def find(data, name):
    hits = sorted(p for p in Path(data).rglob(name) if not p.name.startswith('._'))
    if len(hits) != 1: raise SystemExit(f'expected one {name} under {data}, found {[str(p) for p in hits]}')
    return hits[0]


def read(path, keys, scale, chunk=1024):
    """float32 copy of the H5 arrays divided by scale, as official data.py (np.array(x, float32) / scale), read in chunks."""
    out = {}
    with h5py.File(path, 'r') as h:
        for k in keys:
            d = h[k]; a = np.empty(d.shape, dtype=np.float32)
            for i in range(0, d.shape[0], chunk): a[i:i + chunk] = np.array(d[i:i + chunk], dtype=np.float32) / scale
            out[k] = torch.from_numpy(a)
    return out


class Arrays(Dataset):
    """Returns (gt, pan, ms), the three arrays the official loop uses (batch[0], batch[3], batch[4])."""
    def __init__(self, arrays): self.a = arrays

    def __len__(self): return self.a['gt'].shape[0]

    def __getitem__(self, i): return self.a['gt'][i], self.a['pan'][i], self.a['ms'][i]


# ---------------------------------------------------------------- layer recomputation that keeps BatchNorm statistics exact
RECOMPUTING = {'active': False}


def _bn_save(module, inputs):
    if RECOMPUTING['active']:
        module._saved = (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())


def _bn_restore(module, inputs, output):
    if RECOMPUTING['active'] and getattr(module, '_saved', None) is not None:
        mean, var, count = module._saved
        module.running_mean.copy_(mean); module.running_var.copy_(var); module.num_batches_tracked.copy_(count)
        module._saved = None


def enable_recompute(net):
    """Recompute the six LACRB blocks in backward. State dict keys and outputs are unchanged."""
    for layer in net.layers:
        forward = type(layer).forward

        def wrapped(x, _layer=layer, _forward=forward):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(_forward, _layer, x, use_reentrant=False)
            return _forward(_layer, x)
        layer.forward = wrapped
        for m in layer.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_pre_hook(_bn_save); m.register_forward_hook(_bn_restore)


def backward(loss):
    RECOMPUTING['active'] = True      # every forward that runs inside backward is a recomputation
    try: loss.backward()
    finally: RECOMPUTING['active'] = False


# ---------------------------------------------------------------- training
def seed_all(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    cudnn.benchmark = False; cudnn.deterministic = True


def make(sensor, device, recompute):
    """Same order as official init_train/train: seed, network, (loaders by caller), Adam, StepLR."""
    seed_all(SEED)
    net = adapter.build(SENSORS[sensor][2], sensor, 'retrain', SEED).net.to(device)
    if recompute: enable_recompute(net)
    opt = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': LR}], lr=LR, weight_decay=0)
    sched = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=STEP, gamma=GAMMA, last_epoch=0)
    return net, opt, sched


def validate(net, valid, device, batch=BATCH):
    """Exact mean L1 over the whole validation set (no shuffle, no drop_last), unclipped output as the official loss."""
    net.eval(); total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, valid['gt'].shape[0], batch):
            gt, pan, ms = (valid[k][i:i + batch].to(device) for k in ('gt', 'pan', 'ms'))
            total += torch.abs(net(ms, pan) - gt).sum().item(); n += gt.numel()
    return total / n


def rng_state(device):
    state = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
    if torch.device(device).type == 'cuda': state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['python'])
    if 'cuda' in state and torch.device(device).type == 'cuda': torch.cuda.set_rng_state(state['cuda'], device)


def save(path, payload):
    tmp = Path(str(path) + '.tmp'); torch.save(payload, tmp); os.replace(tmp, path)


def environment(device):
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'cudnn': cudnn.version(), 'numpy': np.__version__, 'host': platform.node()}
    if torch.device(device).type == 'cuda': env['gpu'] = torch.cuda.get_device_name(device)
    return env


def train(a):
    stem, scale, bands = SENSORS[a.sensor]
    epochs = a.epochs or EPOCHS[a.sensor]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    paths = {'train': find(a.data, f'train_{stem}.h5'), 'valid': find(a.data, f'valid_{stem}.h5')}
    files = {k: {'name': p.name, 'bytes': p.stat().st_size, 'sha256': None if a.skip_hash else sha256(p)} for k, p in paths.items()}
    tr = read(paths['train'], ('gt', 'pan', 'ms'), scale); va = read(paths['valid'], ('gt', 'pan', 'ms'), scale)
    if a.limit: tr = {k: v[:a.limit] for k, v in tr.items()}; va = {k: v[:a.limit] for k, v in va.items()}
    print(f'{a.sensor} train {tuple(tr["gt"].shape)} valid {tuple(va["gt"].shape)}', flush=True)

    net, opt, sched = make(a.sensor, a.device, a.recompute)
    loader = DataLoader(Arrays(tr), batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, pin_memory=torch.device(a.device).type == 'cuda')
    config = {'model': 'LAGNet + ADWM (upstream/model/bwnet_lagconv_sota.py BWNet)', 'sensor': a.sensor, 'scale': scale,
              'seed': SEED, 'epochs': epochs, 'epochs_source': 'paper supplement D', 'batch': a.batch, 'loss': 'L1',
              'optimizer': 'Adam lr 2e-3 weight_decay 0', 'scheduler': 'StepLR step 150 gamma 0.5 last_epoch=0 (official)',
              'ifw_hidden': round(48 * adapter.IFW_RATIO[a.sensor]), 'layernorm': 'official (kept)',
              'selection': 'minimum validation L1, every epoch, full validation set', 'recompute': a.recompute,
              'num_workers': a.num_workers, 'files': files, 'limit': a.limit, 'code_commit': a.code_commit,
              'parameters': sum(p.numel() for p in net.parameters())}

    start_epoch, best, history, sessions = 0, float('inf'), [], []
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ck['config']['sensor'] != a.sensor or ck['config']['epochs'] != epochs or ck['config']['batch'] != a.batch: raise SystemExit('resume checkpoint is for another run')
        for k, f in files.items():
            old = ck['config']['files'][k]['sha256']
            if old and f['sha256'] and old != f['sha256']: raise SystemExit(f'{k} data differs from the run: {f["name"]}')
        if ck['config'].get('code_commit') != a.code_commit:
            raise SystemExit(f'resume must use the code commit of the run: {ck["config"].get("code_commit")} (now {a.code_commit})')
        if ck['config']['recompute'] != a.recompute: print('note: recompute setting changed on resume (result unchanged)', flush=True)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        set_rng_state(ck['rng'], a.device)
        start_epoch, best, history, sessions = ck['epoch'], ck['best_validation_l1'], ck['history'], ck['sessions']
        print(f'resumed after epoch {start_epoch}, best validation L1 {best:.7f}', flush=True)
    else:
        save(out / 'initial.pt', {'model': net.state_dict(), 'rng': rng_state(a.device), 'config': config})

    session = {'started': now(), 'start_epoch': start_epoch, 'environment': environment(a.device), 'recompute': a.recompute}
    sessions.append(session)
    t0 = time.time(); deadline = a.deadline_hours * 3600 if a.deadline_hours else None; epoch_seconds = 0.0
    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        e0 = time.time(); lr = opt.param_groups[0]['lr']
        net.train(); losses = []
        for gt, pan, ms in loader:
            gt, pan, ms = gt.to(a.device), pan.to(a.device), ms.to(a.device)
            opt.zero_grad()
            loss = nn.functional.l1_loss(net(ms, pan), gt)
            losses.append(loss.item())
            backward(loss) if a.recompute else loss.backward()
            opt.step()
        sched.step()
        v = validate(net, va, a.device, a.batch)
        improved = v < best; best = min(best, v)
        epoch_seconds = time.time() - e0
        history.append({'epoch': epoch + 1, 'lr': lr, 'train_l1': float(np.nanmean(losses)), 'validation_l1': v,
                        'best': improved, 'seconds': epoch_seconds})
        if improved: save(out / 'best.pt', {'model': net.state_dict(), 'epoch': epoch + 1, 'validation_l1': v, 'config': config})
        session.update(ended=now(), end_epoch=epoch + 1)
        if torch.device(a.device).type == 'cuda': session['max_memory_gb'] = torch.cuda.max_memory_allocated(a.device) / 1024 ** 3
        save(out / 'latest.pt', {'model': net.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                                 'rng': rng_state(a.device), 'epoch': epoch + 1, 'best_validation_l1': best,
                                 'history': history, 'sessions': sessions, 'config': config})
        (out / 'history.json').write_text(json.dumps({'config': config, 'sessions': sessions, 'history': history}, indent=1) + '\n')
        print(f'{a.sensor} epoch {epoch + 1}/{epochs} lr {lr:.2e} train {history[-1]["train_l1"]:.7f} '
              f'valid {v:.7f}{" *" if improved else ""} {epoch_seconds:.0f}s', flush=True)
        if deadline and epoch + 1 < epochs and time.time() - t0 + 1.1 * epoch_seconds > deadline:
            print(f'{a.sensor} DEADLINE: stopped after epoch {epoch + 1}; resume from {out / "latest.pt"}', flush=True)
            break
    done = len(history) == epochs
    status = {'sensor': a.sensor, 'epochs_completed': len(history), 'target_epochs': epochs, 'complete': done,
              'best_epoch': min(history, key=lambda r: r['validation_l1'])['epoch'] if history else None,
              'best_validation_l1': best, 'updated': now()}
    (out / 'status.json').write_text(json.dumps(status, indent=1) + '\n')
    print(f'{a.sensor} {"COMPLETE" if done else "INCOMPLETE"}: {status}', flush=True)


# ---------------------------------------------------------------- measurement before training
def measure(a):
    bands = SENSORS[a.sensor][2]; report = {'sensor': a.sensor, 'environment': environment(a.device), 'runs': []}
    for recompute in (False, True):
        net, opt, _ = make(a.sensor, a.device, recompute); net.train()
        g = torch.Generator().manual_seed(0)
        gt, ms, pan = (torch.rand(s, generator=g).to(a.device) for s in ((BATCH, bands, 64, 64), (BATCH, bands, 16, 16), (BATCH, 1, 64, 64)))
        torch.cuda.reset_peak_memory_stats(a.device)
        try:
            times = []
            for i in range(a.steps + 2):
                torch.cuda.synchronize(a.device); s = time.time()
                opt.zero_grad(); loss = nn.functional.l1_loss(net(ms, pan), gt)
                backward(loss) if recompute else loss.backward(); opt.step()
                torch.cuda.synchronize(a.device); times.append(time.time() - s)
            sec = float(np.mean(times[2:])); net.eval()
            with torch.no_grad():
                torch.cuda.synchronize(a.device); s = time.time(); net(ms, pan); torch.cuda.synchronize(a.device)
            val = (time.time() - s) * VALID_SIZE[a.sensor] / BATCH
            epoch = sec * (TRAIN_SIZE[a.sensor] // BATCH) + val
            report['runs'].append({'recompute': recompute, 'ok': True, 'seconds_per_iteration': sec,
                                   'max_memory_gb': torch.cuda.max_memory_allocated(a.device) / 1024 ** 3,
                                   'estimated_epoch_seconds': epoch, 'estimated_total_hours': epoch * EPOCHS[a.sensor] / 3600})
        except torch.OutOfMemoryError:
            report['runs'].append({'recompute': recompute, 'ok': False, 'error': 'CUDA out of memory at batch 64'})
        del net, opt; torch.cuda.empty_cache()
        print(json.dumps(report['runs'][-1]), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'measure_{a.sensor}.json').write_text(json.dumps(report, indent=1) + '\n')


# ---------------------------------------------------------------- author weights: LayerNorm handling chosen on WV3 validation
def author_select(a):
    path = find(a.data, 'valid_wv3.h5'); va = read(path, ('gt', 'pan', 'ms'), 2047.)
    rows = []
    for mode in adapter.LAYERNORM_MODES:
        net = adapter.build(8, 'WV3', 'author', None, layernorm=mode); adapter.load(net, adapter.AUTHOR_WEIGHTS)
        rows.append({'layernorm': mode, 'validation_l1': validate(net.net.to(a.device), va, a.device)})
        print(rows[-1], flush=True)
    chosen = min(rows, key=lambda r: r['validation_l1'])['layernorm']
    result = {'weights': 'upstream/weights/BWNET_LAGConv_WV3/model.pth', 'weights_sha256': sha256(adapter.AUTHOR_WEIGHTS),
              'validation': {'name': path.name, 'sha256': sha256(path), 'samples': int(va['gt'].shape[0])},
              'metric': 'mean L1 of the unclipped output (DN / 2047), whole validation set', 'rows': rows,
              'chosen': chosen, 'environment': environment(a.device), 'evaluated': now(), 'test_used': False}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'layernorm_selection.json').write_text(json.dumps(result, indent=1) + '\n')
    print(f'chosen layernorm: {chosen}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--sensor', choices=('QB', 'GF2'), required=True)
    t.add_argument('--data', required=True, help='folder searched for train_<s>.h5 and valid_<s>.h5')
    t.add_argument('--out', required=True); t.add_argument('--device', default='cuda')
    t.add_argument('--resume', help='latest.pt of an earlier session'); t.add_argument('--deadline-hours', type=float)
    t.add_argument('--recompute', action='store_true', help='layer recomputation, only if batch 64 does not fit')
    t.add_argument('--num-workers', type=int, default=0)
    t.add_argument('--code-commit', help='CtrS commit of this code, recorded and checked on resume')
    t.add_argument('--epochs', type=int, help='checks only (paper: QB 200, GF2 500); a changed value is not a retrain result')
    t.add_argument('--limit', type=int, help='checks only: first N train and validation samples')
    t.add_argument('--skip-hash', action='store_true', help='checks only: do not hash the data files')
    t.add_argument('--batch', type=int, default=BATCH, help='checks only (paper and official code: 64)')
    m = sub.add_parser('measure'); m.add_argument('--sensor', choices=('QB', 'GF2'), required=True)
    m.add_argument('--device', default='cuda'); m.add_argument('--steps', type=int, default=10); m.add_argument('--out', default='.')
    s = sub.add_parser('author-select'); s.add_argument('--data', required=True); s.add_argument('--device', default='cuda')
    s.add_argument('--out', default=str(HERE / 'results' / 'author_WV3'))
    a = p.parse_args()
    {'train': train, 'measure': measure, 'author-select': author_select}[a.command](a)


if __name__ == '__main__':
    main()
