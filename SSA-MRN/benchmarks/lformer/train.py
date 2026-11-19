"""LFormer retrain on our PanCollection split (QB, GF2), seed 42, weights chosen on validation.

Recipe used: paper (supplementary Sec. 1, AGENTS.md 8; issue #184). code is kept for reference only (README.md, 방법):
    code   official configs/config_<sensor>.yaml + main.py/pipeline.py: 500 epochs, batch 32, Adam lr 1e-3, weight decay 1e-6,
           StepLR(250, 0.1)
    paper  supplementary Sec. 1 (= arXiv 2404.12804v1 Sec. 4.1): 800 epochs, batch 32, Adam (0.9, 0.999) lr 3e-4, weight decay 0.1,
           x0.1 at epochs 300 and 500
Both: model sizes from the official config (the paper gives none), loss L1 + 0.1 (1 - SSIM) (official get_loss('l1ssim'),
paper Eq. 15-17), shuffle + drop_last, cudnn.deterministic (official setup_seed).
Memory: batch 32 of 64x64 patches keeps about 7 GB (QB) / 10 GB (GF2) of activations (CPU measurement, GPU differs); if it
does not fit, --checkpointing recomputes each stage in the backward pass (same batch, loss and gradients; slower) and is
recorded in history.json meta.checkpointing. Never lower the batch size.

Changes required by CtrS rules (not training changes): seed 42 (official 3407); validation on valid_<sensor>.h5 every epoch
(official code validates on the training file and keeps every 25th epoch); best.pt = lowest validation loss (official
criterion, unclipped output as official validate()); resume from latest.pt.

Outputs in --out (default checkpoints/retrain_<sensor>_s<seed>/):
    best.pt      {'model', 'epoch', 'val_loss', 'meta'}                Git
    latest.pt    model, optimizer, scheduler, RNG, epoch, best          Release (common/release_weights.py), for --resume
    initial.pt   weights before the first step                         Release
    history.json per-epoch train loss, validation loss / L1, lr, seconds -> results/retrain_<sensor>_s<seed>/history.json

python SSA-MRN/benchmarks/lformer/train.py --sensor QB --recipe paper --train data/train_qb.h5 --val data/valid_qb.h5 --device cuda
python SSA-MRN/benchmarks/lformer/train.py ... --resume                    # continue from <out>/latest.pt
python SSA-MRN/benchmarks/lformer/train.py ... --limit 256 --epochs 2 --out <tmp>   # smoke test
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
SCALE = {'QB': 2047.0, 'GF2': 1023.0, 'WV3': 2047.0}
BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8}
RECIPES = {
    'code': dict(epochs=500, batch_size=32, lr={'QB': 1e-3, 'GF2': 1e-3, 'WV3': 3e-4}, weight_decay=1e-6,
                 milestones=[250], gamma=0.1, source='official configs/config_<sensor>.yaml, models/pipeline.py (StepLR 250)'),
    'paper': dict(epochs=800, batch_size=32, lr={'QB': 3e-4, 'GF2': 3e-4, 'WV3': 3e-4}, weight_decay=0.1,
                  milestones=[300, 500], gamma=0.1, source='paper supplementary Sec. 1 (= arXiv v1 Sec. 4.1)'),
}
KST = timezone(timedelta(hours=9))
SETTINGS = ('sensor', 'recipe', 'seed', 'epochs', 'batch_size', 'lr', 'weight_decay', 'milestones', 'gamma',
            'train_sha256', 'val_sha256', 'limit')


def adapter():
    spec = importlib.util.spec_from_file_location('lformer_adapter', HERE / 'model.py')
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
        data = {k: torch.from_numpy(np.asarray(h[k][:n], dtype=np.float32) / SCALE[sensor]) for k in ('pan', 'lms', 'gt')}
    if data['gt'].shape[1] != BANDS[sensor]: raise SystemExit(f'{path}: {data["gt"].shape[1]} bands, {sensor} has {BANDS[sensor]}')
    if data['gt'].shape[-1] % 8 or data['gt'].shape[-2] % 8: raise SystemExit(f'{path}: patch size must be a multiple of 8')
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
    return tuple(data[k][idx].to(device, non_blocking=True) for k in ('pan', 'lms', 'gt'))


def validate(net, criterion, data, device, size=32):
    """Official criterion and L1 on the unclipped output, averaged over samples (last partial batch weighted by size)."""
    net.train(False); total, l1, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, data['gt'].shape[0], size):
            pan, lms, gt = batch(data, torch.arange(i, min(i + size, data['gt'].shape[0])), device)
            out = net.net._forward_implem(lms, pan) + lms
            loss, _ = criterion(out, gt); b = gt.shape[0]
            total += loss.item() * b; l1 += (out - gt).abs().mean().item() * b; count += b
    return total / count, l1 / count


def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sensor', required=True, choices=('QB', 'GF2', 'WV3'))
    p.add_argument('--recipe', required=True, choices=tuple(RECIPES), help='paper (used: supplementary Sec. 1); code = official config, reference only')
    p.add_argument('--train', required=True, type=Path); p.add_argument('--val', required=True, type=Path)
    p.add_argument('--epochs', type=int, help='smoke tests only; default: recipe')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, help='default checkpoints/retrain_<sensor>_s<seed>')
    p.add_argument('--resume', action='store_true', help='continue from <out>/latest.pt')
    p.add_argument('--deadline', type=float, help='unix time; stop after the last epoch that ends before it (Kaggle)')
    p.add_argument('--limit', type=int, help='smoke test: first N training and validation samples')
    p.add_argument('--checkpointing', action='store_true', help='recompute stages in backward (memory; same gradients, slower)')
    a = p.parse_args()
    if a.train.resolve() == a.val.resolve(): p.error('training and validation must be different files')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA is not available')
    r = RECIPES[a.recipe]; epochs = a.epochs or r['epochs']; lr = r['lr'][a.sensor]
    out = a.out or HERE / 'checkpoints' / f'retrain_{a.sensor}_s{a.seed}'
    out.mkdir(parents=True, exist_ok=True); hist_path = out / 'history.json'

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True     # official utils.setup_seed

    t_load = time.time()
    train, val = read(a.train, a.sensor, a.limit), read(a.val, a.sensor, a.limit)
    n = train['gt'].shape[0]
    if n < r['batch_size']: p.error(f'{n} training samples < batch {r["batch_size"]}')
    mod = adapter()
    net = mod.build(BANDS[a.sensor], a.sensor, 'retrain', a.seed).to(a.device)
    criterion = mod.official('loss_utils').get_loss('l1ssim').to(a.device)      # L1 + 0.1 (1 - SSIM)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=r['weight_decay'])   # official: betas default
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=r['milestones'], gamma=r['gamma'])
    meta = dict(model='lformer', setting='retrain', sensor=a.sensor, recipe=a.recipe, recipe_source=r['source'], seed=a.seed,
                epochs=epochs, batch_size=r['batch_size'], lr=lr, weight_decay=r['weight_decay'], milestones=r['milestones'],
                gamma=r['gamma'], loss='L1 + 0.1 (1 - SSIM), official get_loss("l1ssim")', optimizer='Adam (0.9, 0.999)',
                dims=dict(zip(('lms_dim', 'attn_dim', 'hp_dim'), mod.DIMS[a.sensor])), n_stage=mod.N_STAGE,
                selection='lowest validation loss (official criterion, unclipped), every epoch',
                parameters=sum(q.numel() for q in net.parameters()),
                train=a.train.name, train_sha256=sha256(a.train), train_samples=n, val=a.val.name, val_sha256=sha256(a.val),
                val_samples=val['gt'].shape[0], limit=a.limit, device=a.device, torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None, checkpointing=a.checkpointing)
    g = torch.Generator().manual_seed(a.seed)
    start, best, history = 0, float('inf'), []
    if a.resume:
        ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if {k: ck['meta'][k] for k in SETTINGS} != {k: meta[k] for k in SETTINGS}: p.error('latest.pt was made with other settings or data')
        net.net.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        set_rng_state(ck['rng']); g.set_state(ck['shuffle_rng']); start, best = ck['epoch'], ck['best_val_loss']
        history = [h for h in json.loads(hist_path.read_text(encoding='utf-8'))['epochs'] if h['epoch'] <= start]
        meta['sessions'] = ck['meta'].get('sessions', 1) + 1
        meta['session_log'] = ck['meta'].get('session_log', [])
    else:
        atomic_save(dict(model=cpu(net.net.state_dict()), epoch=0, meta=meta), out / 'initial.pt')
        meta['sessions'], meta['session_log'] = 1, []
    # one entry per run: where each part of the training ran (Kaggle / Colab, GPU, torch); results are not bit-identical across GPUs
    meta['session_log'].append(dict(session=meta['sessions'], start_epoch=start, gpu=meta['gpu'], torch=meta['torch'],
                                    checkpointing=a.checkpointing, platform=os.environ.get('LFORMER_PLATFORM', 'unknown'),
                                    started=datetime.now(KST).isoformat(timespec='seconds')))
    if start >= epochs: print(f'already finished ({start}/{epochs} epochs)'); return
    net.set_checkpointing(a.checkpointing)
    print(f'{a.sensor} recipe {a.recipe}: {n} train / {val["gt"].shape[0]} val samples loaded in {time.time() - t_load:.0f}s; '
          f'epochs {start}->{epochs}; lr {lr:g}; {meta["parameters"]} parameters', flush=True)

    bs = r['batch_size']
    for epoch in range(start, epochs):
        t0 = time.time(); net.train(); losses = []
        order = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):                                  # shuffle, drop_last (official DataLoader)
            pan, lms, gt = batch(train, order[i:i + bs], a.device)
            optimizer.zero_grad()
            loss, _ = criterion(net(pan, lms), gt)                           # official train_step: loss on lms + residual
            loss.backward(); optimizer.step(); losses.append(loss.item())
        cur_lr = optimizer.param_groups[0]['lr']; scheduler.step()
        val_loss, val_l1 = validate(net, criterion, val, a.device)
        row = dict(epoch=epoch + 1, train_loss=float(np.nanmean(losses)), val_loss=val_loss, val_l1=val_l1, lr=cur_lr,
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        if val_loss < best:
            best = val_loss
            atomic_save(dict(model=cpu(net.net.state_dict()), epoch=epoch + 1, val_loss=val_loss, meta=meta), out / 'best.pt')
        atomic_save(dict(model=cpu(net.net.state_dict()), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         rng=rng_state(), shuffle_rng=g.get_state(), epoch=epoch + 1, best_val_loss=best, meta=meta), out / 'latest.pt')
        best_row = min(history, key=lambda h: h['val_loss'])
        hist_path.write_text(json.dumps(dict(meta=meta, best_epoch=best_row['epoch'], best_val_loss=best_row['val_loss'],
                                             updated=datetime.now(KST).isoformat(timespec='seconds'), epochs=history),
                                        ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f"{a.sensor} epoch {epoch + 1}/{epochs} train {row['train_loss']:.6f} val {val_loss:.6f} (L1 {val_l1:.6f}) "
              f"lr {cur_lr:.2e} {row['seconds']}s" + (' *best' if val_loss == best else ''), flush=True)
        if a.deadline and epoch + 1 < epochs and time.time() + 1.3 * row['seconds'] + 60 > a.deadline:
            print(f'{a.sensor}: deadline, stopped after epoch {epoch + 1}; continue with --resume', flush=True); break


if __name__ == '__main__':
    main()
