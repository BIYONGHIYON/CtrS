"""Train ADWM (official LAGNet + ADWM) on our PanCollection split: setting retrain, QB and GF2.

Official main_train.py and the paper (Sec. 3.1, App. D), with the CtrS rules for seed and selection:
    L1 loss, Adam (weight decay 0), batch 64, lr 2e-3 halved every 150 epochs. The official scheduler is built with
    last_epoch=0, so the first halving comes after 149 epochs (then 299, 449); this script builds it the same way.
    epochs: QB 200 with CACW d = 0.6n; GF2 as WV3, 500 with d = 0.8n (paper App. D). Official main_train.py default: 500.
    seed 42 (CtrS; official default 1), cudnn deterministic, benchmark off (as main_train.py)
    validation L1 every epoch (official: every 10), best.pt = lowest validation L1 (CtrS: select with validation only)
Inputs are DN / scale (QB 2047, GF2 1023, as the official data.py and common/data.py). The H5 arrays are read into memory.
Memory: --checkpointing auto first tries one real batch of 64 without recomputation and turns on LAConv recomputation
(model.ADWM.set_checkpointing) only if it does not fit; the batch, the loss and the BatchNorm statistics stay those of 64.
Sessions: training may be split over several sessions (--deadline, --resume). Every start appends one entry to
meta.session_log (session, platform, GPU, LAConv recomputation, torch/CUDA, start and end epoch, times), so the report can
state where each part of the training ran. --deadline stops before an epoch that would not end in time, estimated from the
slowest of the last three epochs (the first epoch of a new run starts unless the deadline has already passed).

Outputs in --out (default checkpoints/retrain_<sensor>_s<seed>/):
    best.pt      {'model', 'epoch', 'val_l1', 'meta'}            Git (benchmarks checkpoints are tracked)
    latest.pt    model, optimizer, scheduler, RNG, epoch, best   Release (common/release_weights.py), for --resume
    initial.pt   weights before the first step                  Release
    history.json per-epoch train L1, validation L1, lr, seconds, and meta (with session_log); written before latest.pt.
                 provenance/session.py copies it to results/retrain_<sensor>_s<seed>/history.json when the run is evaluated.

python SSA-MRN/benchmarks/adwm/train.py --sensor QB --train data/train_qb.h5 --val data/valid_qb.h5 --device cuda
python SSA-MRN/benchmarks/adwm/train.py ... --resume                                # continue from <out>/latest.pt
python SSA-MRN/benchmarks/adwm/train.py ... --deadline <unix time>                  # stop before a Kaggle session ends
python SSA-MRN/benchmarks/adwm/train.py ... --session 3                             # session number for meta.session_log
python SSA-MRN/benchmarks/adwm/train.py ... --epochs 2 --limit 128 --out /tmp/adwm_smoke   # smoke test
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EPOCHS = {'QB': 200, 'GF2': 500}
SCALE = {'QB': 2047., 'GF2': 1023.}
BANDS = {'QB': 4, 'GF2': 4}
KST = timezone(timedelta(hours=9))
SETTINGS = ('sensor', 'seed', 'epochs', 'batch_size', 'lr', 'step_size', 'gamma', 'cacw_ratio', 'train_sha256', 'val_sha256', 'limit')


def adapter():
    spec = importlib.util.spec_from_file_location('adwm_adapter', HERE / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path, sensor, limit):
    with h5py.File(path, 'r') as h:
        n = h['gt'].shape[0] if limit is None else min(limit, h['gt'].shape[0])
        data = {k: torch.from_numpy(np.asarray(h[k][:n], dtype=np.float32) / SCALE[sensor]) for k in ('pan', 'ms', 'gt')}
    if data['gt'].shape[1] != BANDS[sensor]: raise SystemExit(f'{path}: {data["gt"].shape[1]} bands, {sensor} has {BANDS[sensor]}')
    if not all(torch.isfinite(v).all() for v in data.values()): raise SystemExit(f'{path}: nonfinite values')
    return data


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])


def cpu(state): return {k: v.detach().cpu().clone() for k, v in state.items()}


def batch(data, idx, device):
    return (data['ms'][idx].to(device, non_blocking=True), data['pan'][idx].to(device, non_blocking=True),
            data['gt'][idx].to(device, non_blocking=True))


def needs_checkpointing(net, data, size, device):
    """One forward/backward of a real batch without recomputation; the model, BatchNorm statistics and RNG are restored."""
    if not str(device).startswith('cuda'): return False
    state, rng = cpu(net.state_dict()), rng_state()
    try:
        net.train(); net.zero_grad(set_to_none=True)
        ms, pan, gt = batch(data, torch.arange(size), device)
        torch.nn.functional.l1_loss(net.net(ms, pan), gt).backward()
        return False
    except torch.OutOfMemoryError:
        return True
    finally:
        net.zero_grad(set_to_none=True); net.load_state_dict(state); set_rng_state(rng); torch.cuda.empty_cache()


def validate(net, data, device, size=64):
    net.eval(); total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, data['gt'].shape[0], size):
            ms, pan, gt = batch(data, torch.arange(i, min(i + size, data['gt'].shape[0])), device)
            total += (net.net(ms, pan) - gt).abs().sum().item(); count += gt.numel()   # unclamped, as the official validation
    return total / count


def platform():
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or Path('/kaggle').is_dir(): return 'kaggle'
    if os.environ.get('COLAB_RELEASE_TAG'): return 'colab'
    return 'local'


def epoch_estimate(history):
    """Seconds the next epoch may take: the slowest of the last three epochs (None before the first epoch)."""
    recent = [h['seconds'] for h in history[-3:]]
    return max(recent) if recent else None


def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sensor', required=True, choices=tuple(EPOCHS))
    p.add_argument('--train', required=True, type=Path); p.add_argument('--val', required=True, type=Path)
    p.add_argument('--epochs', type=int, help='default: paper (QB 200, GF2 500)')
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--step-size', type=int, default=150); p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda')
    p.add_argument('--checkpointing', choices=('auto', 'on', 'off'), default='auto', help='LAConv recomputation (memory)')
    p.add_argument('--out', type=Path, help='default checkpoints/retrain_<sensor>_s<seed>')
    p.add_argument('--resume', action='store_true', help='continue from <out>/latest.pt')
    p.add_argument('--deadline', type=float, help='unix time; stop after the last epoch that ends before it')
    p.add_argument('--limit', type=int, help='smoke test: first N training and validation samples')
    p.add_argument('--session', type=int, help='session number (provenance/session.py), recorded in meta.session_log')
    a = p.parse_args()
    if a.train.resolve() == a.val.resolve(): p.error('training and validation must be different files')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA is not available')
    epochs = a.epochs or EPOCHS[a.sensor]
    out = a.out or HERE / 'checkpoints' / f'retrain_{a.sensor}_s{a.seed}'
    out.mkdir(parents=True, exist_ok=True); hist_path = out / 'history.json'

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True     # as main_train.py

    t_load = time.time()
    train, val = read(a.train, a.sensor, a.limit), read(a.val, a.sensor, a.limit)
    n = train['gt'].shape[0]
    if n < a.batch_size: p.error(f'{n} training samples < batch {a.batch_size}')
    mod = adapter()
    net = mod.build(BANDS[a.sensor], a.sensor, 'retrain', a.seed).to(a.device)
    # official main_train.py: Adam with initial_lr and StepLR(last_epoch=0) -> halvings after epochs 149, 299, 449
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': a.lr}], lr=a.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=a.step_size, gamma=a.gamma, last_epoch=0)
    meta = dict(model='adwm', setting='retrain', sensor=a.sensor, seed=a.seed, epochs=epochs, batch_size=a.batch_size, lr=a.lr,
                step_size=a.step_size, gamma=a.gamma, scheduler='StepLR(last_epoch=0) as official main_train.py', loss='L1',
                cacw_ratio=net.cacw_ratio, selection='lowest validation L1, every epoch', parameters=sum(q.numel() for q in net.parameters()),
                train=a.train.name, train_sha256=sha256(a.train), train_samples=n, val=a.val.name, val_sha256=sha256(a.val),
                val_samples=val['gt'].shape[0], limit=a.limit)
    g = torch.Generator().manual_seed(a.seed)
    start, best, history, log = 0, float('inf'), [], []
    if a.resume:
        ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if {k: ck['meta'][k] for k in SETTINGS} != {k: meta[k] for k in SETTINGS}: p.error('latest.pt was made with other settings or data')
        net.net.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        set_rng_state(ck['rng']); g.set_state(ck['shuffle_rng']); start, best = ck['epoch'], ck['best_val_l1']
        history = [h for h in json.loads(hist_path.read_text(encoding='utf-8'))['epochs'] if h['epoch'] <= start]
        log = list(ck['meta'].get('session_log', []))
    else:
        atomic_save(dict(model=cpu(net.net.state_dict()), epoch=0, meta=meta), out / 'initial.pt')
    if start >= epochs: print(f'already finished ({start}/{epochs} epochs)'); return

    ck_on = a.checkpointing == 'on' or (a.checkpointing == 'auto' and needs_checkpointing(net, train, a.batch_size, a.device))
    net.set_checkpointing(ck_on)
    entry = dict(session=a.session, run=len(log) + 1, platform=platform(), device=a.device,
                 gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,
                 checkpointing=ck_on, checkpointing_mode=a.checkpointing, torch=str(torch.__version__), cuda=str(torch.version.cuda),
                 start_epoch=start, end_epoch=start, started=datetime.now(KST).isoformat(timespec='seconds'), updated=None)
    meta['session_log'] = log + [entry]
    print(f'{a.sensor}: {n} train / {val["gt"].shape[0]} val samples loaded in {time.time() - t_load:.0f}s; epochs {start}->{epochs}; '
          f'LAConv recomputation {"on" if ck_on else "off"}; {meta["parameters"]} parameters', flush=True)

    for epoch in range(start, epochs):
        est = epoch_estimate(history)
        if a.deadline and (time.time() > a.deadline or (est and time.time() + 1.3 * est + 60 > a.deadline)):
            print(f'{a.sensor}: deadline, stopped before epoch {epoch + 1} (about {est:.0f}s per epoch); continue with --resume', flush=True)
            break
        t0 = time.time(); net.train(); losses = []
        order = torch.randperm(n, generator=g)
        for i in range(0, n - a.batch_size + 1, a.batch_size):          # shuffle, drop_last (as the official DataLoader)
            ms, pan, gt = batch(train, order[i:i + a.batch_size], a.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(net.net(ms, pan), gt)
            loss.backward(); optimizer.step(); losses.append(loss.item())
        lr = optimizer.param_groups[0]['lr']; scheduler.step()
        val_l1 = validate(net, val, a.device)
        row = dict(epoch=epoch + 1, train_l1=float(np.mean(losses)), val_l1=val_l1, lr=lr, seconds=round(time.time() - t0, 1))
        history.append(row)
        entry.update(end_epoch=epoch + 1, updated=datetime.now(KST).isoformat(timespec='seconds'))
        if val_l1 < best:
            best = val_l1
            atomic_save(dict(model=cpu(net.net.state_dict()), epoch=epoch + 1, val_l1=val_l1, meta=meta), out / 'best.pt')
        # history first: if the session dies between the two writes, --resume drops the extra row (epoch > latest)
        best_row = min(history, key=lambda h: h['val_l1'])
        tmp = hist_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(meta=meta, best_epoch=best_row['epoch'], best_val_l1=best_row['val_l1'],
                                       updated=entry['updated'], epochs=history), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        tmp.replace(hist_path)
        atomic_save(dict(model=cpu(net.net.state_dict()), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         rng=rng_state(), shuffle_rng=g.get_state(), epoch=epoch + 1, best_val_l1=best, meta=meta), out / 'latest.pt')
        print(f"{a.sensor} epoch {epoch + 1}/{epochs} train {row['train_l1']:.6f} val {val_l1:.6f} lr {lr:.2e} {row['seconds']}s"
              + (' *best' if val_l1 == best else ''), flush=True)


if __name__ == '__main__':
    main()
