"""Train ADWM (official LAGNet + ADWM, upstream/model/bwnet_lagconv_sota.py) on our PanCollection split (setting retrain).

Follows the official main_train.py and the paper's implementation details, with the CtrS rules for seeds and selection:
    loss L1, Adam (weight decay 0), lr 2e-3 halved every 150 epochs (StepLR), batch 64 (official main_train.py)
    epochs: QB 200, GF2 500, WV3 500 (paper implementation details; main_train.py default is 500 for every sensor)
    seed 42 (CtrS rule; official default 1), cudnn deterministic as in main_train.py
    validation L1 every epoch on the validation H5, best.pt = lowest validation L1 (CtrS rule: select with validation only;
    the official script validates every 10 epochs and keeps every 10th epoch instead)
    gradient accumulation: batch 64 needs about 19 GB of activations (paper: RTX 3090, 24 GB), more than a Kaggle T4 (15 GB).
    --accum N splits every batch of 64 into N equal chunks, back-propagates each (loss / N) and steps once, so the loss and the
    update are those of the full batch; only BatchNorm sees N-times smaller batches. --accum auto (default) tries 1, 2, 4, 8
    on one real batch before training and keeps the smallest that fits; the choice is stored in history.json and latest.pt.
Inputs are DN / sensor scale (QB, WV3 2047; GF2 1023), the same as common/data.py and the evaluator.

Outputs in --out (default SSA-MRN/benchmarks/adwm/checkpoints/retrain_<sensor>_s<seed>/):
    best.pt     {'model': state_dict, 'epoch', 'val_l1', ...}   kept in Git (git add -f), loaded by model.load
    latest.pt   model + optimizer + scheduler + RNG states for --resume  -> GitHub Release, not Git
    initial.pt  weights before the first step                          -> GitHub Release, not Git
    history.json  per-epoch train L1, validation L1, lr, seconds        -> results/retrain_<sensor>_s<seed>/history.json

python SSA-MRN/benchmarks/adwm/train.py --sensor QB --train data/train_qb.h5 --val data/valid_qb.h5 --device cuda
python SSA-MRN/benchmarks/adwm/train.py --sensor QB --train ... --val ... --device cuda --resume   # continue from latest.pt
python SSA-MRN/benchmarks/adwm/train.py ... --deadline $(( $(date +%s) + 39000 ))   # stop before a Kaggle 12 h session ends
python SSA-MRN/benchmarks/adwm/train.py --sensor QB --train ... --val ... --epochs 1 --limit 64 --out /tmp/adwm_smoke  # smoke
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from common.data import PanCollectionH5  # noqa: E402

EPOCHS = {'QB': 200, 'GF2': 500, 'WV3': 500}       # paper implementation details
BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8}
KST = timezone(timedelta(hours=9))


def adapter():
    spec = importlib.util.spec_from_file_location('adwm_adapter', HERE / 'model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cpu(state):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s.get('cuda') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])


def choose_accum(value, net, dataset, batch, device):
    if value != 'auto':
        n = int(value)
        if batch % n: raise SystemExit(f'batch {batch} is not divisible by --accum {n}')
        return n
    if not str(device).startswith('cuda'): return 1
    state = {k: v.clone() for k, v in net.state_dict().items()}; rng = rng_state()   # the probe must not change the run
    idx = list(range(min(batch, len(dataset)))); b = [dataset[i] for i in idx]
    ms = torch.stack([x['ms'] for x in b]).to(device); pan = torch.stack([x['pan'] for x in b]).to(device)
    try:
        for n in (1, 2, 4, 8):
            if batch % n: continue
            try:
                net.train(True); net.zero_grad(set_to_none=True)
                net.net(ms[:batch // n], pan[:batch // n]).mean().backward()
                return n
            except torch.OutOfMemoryError:
                print(f'accum {n} (micro-batch {batch // n}) does not fit in GPU memory', flush=True)
            finally:
                net.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        raise SystemExit('even micro-batch 8 does not fit')
    finally:
        net.load_state_dict(state); set_rng_state(rng)


def validate(net, loader, device):
    net.train(False); total, count = 0.0, 0
    with torch.no_grad():
        for b in loader:
            pan, ms, gt = b['pan'].to(device), b['ms'].to(device), b['gt'].to(device)
            out = net.net(ms, pan)               # unclamped, as the official validation loss
            total += torch.nn.functional.l1_loss(out, gt, reduction='sum').item(); count += gt.numel()
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sensor', required=True, choices=tuple(EPOCHS))
    p.add_argument('--train', required=True, type=Path); p.add_argument('--val', required=True, type=Path)
    p.add_argument('--epochs', type=int, help='default: paper value (QB 200, GF2 500, WV3 500)')
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--step-size', type=int, default=150); p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--accum', default='auto', help='chunks per batch (1, 2, 4, 8) or auto: smallest that fits in GPU memory')
    p.add_argument('--out', type=Path, help='default checkpoints/retrain_<sensor>_s<seed>')
    p.add_argument('--resume', action='store_true', help='continue from <out>/latest.pt')
    p.add_argument('--limit', type=int, help='smoke test: first N training and validation samples')
    p.add_argument('--stop-after', type=int, help='exit after this many epochs in this session (resume later with --resume)')
    p.add_argument('--deadline', type=float, help='unix time; stop after the epoch that would otherwise run past it (Kaggle 12 h)')
    a = p.parse_args()
    epochs = a.epochs or EPOCHS[a.sensor]
    out = a.out or HERE / 'checkpoints' / f'retrain_{a.sensor}_s{a.seed}'
    if a.train.resolve() == a.val.resolve(): p.error('training and validation must be different H5 files')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA is not available')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True   # as main_train.py

    train_set = PanCollectionH5(a.train, a.sensor, limit=a.limit); val_set = PanCollectionH5(a.val, a.sensor, limit=a.limit)
    if train_set.channels != BANDS[a.sensor] or val_set.channels != BANDS[a.sensor]: p.error('band count does not match sensor')
    g = torch.Generator(); g.manual_seed(a.seed)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                              pin_memory=a.device.startswith('cuda'), generator=g)
    val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)

    net = adapter().build(BANDS[a.sensor], a.sensor, 'retrain', a.seed).to(a.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=a.step_size, gamma=a.gamma)
    accum = choose_accum(a.accum, net, train_set, a.batch_size, a.device)
    print(f'batch {a.batch_size} = {accum} x micro-batch {a.batch_size // accum}', flush=True)
    out.mkdir(parents=True, exist_ok=True); hist_path = out / 'history.json'
    meta = dict(model='adwm', setting='retrain', sensor=a.sensor, seed=a.seed, epochs=epochs, batch_size=a.batch_size,
                lr=a.lr, step_size=a.step_size, gamma=a.gamma, loss='L1', selection='validation L1 minimum, every epoch',
                accum=accum, micro_batch=a.batch_size // accum,
                train=a.train.name, val=a.val.name, limit=a.limit, device=a.device,
                gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None, torch=torch.__version__)
    start, best, history = 0, float('inf'), []
    if a.resume:
        ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        keys = ('sensor', 'seed', 'epochs', 'batch_size', 'lr', 'accum')
        if a.accum == 'auto' and 'accum' in ck['meta']: meta['accum'] = ck['meta']['accum']; meta['micro_batch'] = a.batch_size // meta['accum']; accum = meta['accum']
        if {k: ck['meta'].get(k) for k in keys} != {k: meta[k] for k in keys}:
            p.error('latest.pt was made with different settings')
        net.net.load_state_dict(ck["model"]); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        set_rng_state(ck['rng']); g.set_state(ck['loader_rng']); start, best = ck['epoch'], ck['best_val_l1']
        history = json.loads(hist_path.read_text(encoding='utf-8'))['epochs'] if hist_path.exists() else []
        history = [h for h in history if h['epoch'] <= start]
    else:
        torch.save(dict(model=cpu(net.net.state_dict()), epoch=0, meta=meta), out / 'initial.pt')

    stop = epochs if a.stop_after is None else min(epochs, start + a.stop_after)
    for epoch in range(start, stop):
        net.train(True); t0 = time.time(); losses = []
        for b in train_loader:
            pan, ms, gt = b['pan'].to(a.device, non_blocking=True), b['ms'].to(a.device, non_blocking=True), b['gt'].to(a.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); total = 0.0
            for m_, p_, g_ in zip(ms.chunk(accum), pan.chunk(accum), gt.chunk(accum)):
                loss = torch.nn.functional.l1_loss(net.net(m_, p_), g_) / accum   # equal chunks: sum = full-batch mean
                loss.backward(); total += loss.item()
            optimizer.step(); losses.append(total)
        lr = optimizer.param_groups[0]['lr']; scheduler.step()
        val = validate(net, val_loader, a.device)
        row = dict(epoch=epoch + 1, train_l1=float(np.mean(losses)), val_l1=val, lr=lr, seconds=round(time.time() - t0, 1))
        history.append(row)
        improved = val < best
        if improved:
            best = val
            torch.save(dict(model=cpu(net.net.state_dict()), epoch=epoch + 1, val_l1=val, meta=meta), out / 'best.pt')
        torch.save(dict(model=cpu(net.net.state_dict()), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng=rng_state(), loader_rng=g.get_state(), epoch=epoch + 1, best_val_l1=best, meta=meta), out / 'latest.tmp')
        (out / 'latest.tmp').replace(out / 'latest.pt')   # never leave a half-written latest.pt if the session is killed
        hist_path.write_text(json.dumps(dict(meta=meta, best_epoch=min(history, key=lambda h: h['val_l1'])['epoch'],
                                             updated=datetime.now(KST).isoformat(timespec='seconds'), epochs=history),
                                        ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f"epoch {epoch + 1}/{epochs} train_l1 {row['train_l1']:.6f} val_l1 {val:.6f} lr {lr:.2e} {row['seconds']}s"
              + (' *best' if improved else ''), flush=True)
        if a.deadline and epoch + 1 < stop and time.time() + 1.5 * row['seconds'] > a.deadline:
            print(f'deadline: stopping after epoch {epoch + 1} (continue with --resume)', flush=True); break


if __name__ == '__main__':
    main()
