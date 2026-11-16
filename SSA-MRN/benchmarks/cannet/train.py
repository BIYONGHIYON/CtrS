"""Retrain CANNet on our PanCollection split with the official training recipe (setting `retrain`).

Official pieces used unchanged (upstream/, Duanyll/CANConv): CANNet, H5PanDataset, seed_everything, the k-means cache and its
KMeansCacheScheduler. Recipe from canconv/models/cannet/default.json + presets.json (qb/gf2/wv3): L1 loss, Adam (weight decay 0),
batch 32, 500 epochs, StepLR, k-means cache policy [[10, 1], [100, 10], [300, 20], 50], validation every 10 epochs.

Added for our protocol (the official trainer saves fixed epochs and never selects): the seed is ours (42, 43, 44); the weights with the
lowest validation L1 (the official criterion, evaluated every 10 epochs as in the official config) become best.pt; latest.pt holds
everything needed to resume (model, optimizer, scheduler, RNG states, k-means scheduler epoch). The k-means cache itself (a cluster
map for every training patch) is too large to store, so the first epoch after a resume rebuilds it from scratch (cache mode "init")
and the official policy continues from the next epoch. --deadline stops cleanly before a time limit (Kaggle 12 h sessions).

python train.py --sensor QB --seed 42 --data /path/with/h5 --out runs --deadline 1763300000
python train.py --sensor QB --seed 42 --data /path/with/h5 --out runs --resume runs/retrain_QB_s42/latest.pt
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import Adapter  # noqa: E402  (also puts upstream/ on sys.path)
from canconv.dataset.h5pan import H5PanDataset  # noqa: E402
from canconv.layers.kmeans import KMeansCacheScheduler, reset_cache  # noqa: E402
from canconv.layers.kmeans import cache as kmeans_cache  # noqa: E402
from canconv.util.seed import seed_everything  # noqa: E402

# official recipe: default.json, overridden per sensor by presets.json
RECIPE = dict(batch_size=32, epochs=500, learning_rate=1e-3, lr_step_size=250, loss='l1', val_interval=10,
              kmeans_cache_update=[[10, 1], [100, 10], [300, 20], 50])
PRESETS = {'QB': dict(stem='qb', bands=4, scale=2047., learning_rate=5e-4, lr_step_size=500),
           'GF2': dict(stem='gf2', bands=4, scale=1023., learning_rate=5e-4, lr_step_size=500),
           'WV3': dict(stem='wv3', bands=8, scale=2047.)}


def find(data, name):
    """First file called `name` under data (Kaggle flattens folder names into the file name, so match the suffix)."""
    hits = sorted(p for p in Path(data).rglob('*') if p.is_file() and (p.name == name or p.name.endswith('__' + name))
                  and not p.name.endswith('._' + name) and p.stat().st_size > 1 << 20)
    if not hits: raise SystemExit(f'{name} not found under {data}')
    return hits[0]


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


@torch.no_grad()
def validate(net, loader, dev):
    net.eval(); l1 = mse = n = 0.
    for b in loader:
        sr = net(b['pan'].to(dev), b['lms'].to(dev)); gt = b['gt'].to(dev); k = gt.shape[0]
        l1 += torch.nn.functional.l1_loss(sr, gt).item() * k; mse += torch.nn.functional.mse_loss(sr, gt).item() * k; n += k
    return l1 / n, mse / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sensor', choices=tuple(PRESETS), default='QB')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data', required=True, help='folder containing train_<s>.h5 and valid_<s>.h5 (searched recursively)')
    p.add_argument('--out', default='runs')
    p.add_argument('--resume', help='latest.pt to continue from')
    p.add_argument('--epochs', type=int, default=RECIPE['epochs'])
    p.add_argument('--deadline', type=float, help='unix time; stop with latest.pt saved before the next epoch would pass it')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cfg = RECIPE | PRESETS[a.sensor]; dev = torch.device(a.device)
    if dev.type == 'cuda' and dev.index: torch.cuda.set_device(dev)
    out = Path(a.out) / f'retrain_{a.sensor}_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    log = (out / 'train.log').open('a', encoding='utf-8')

    def say(*x):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(map(str, x)); print(line, flush=True); log.write(line + '\n'); log.flush()

    seed_everything(a.seed)  # official helper (also disables cuDNN, as in the official trainer)
    train_file, val_file = find(a.data, f"train_{cfg['stem']}.h5"), find(a.data, f"valid_{cfg['stem']}.h5")
    train_set, val_set = H5PanDataset(str(train_file), scale=cfg['scale']), H5PanDataset(str(val_file), scale=cfg['scale'])
    say('data', train_file.name, len(train_set), val_file.name, len(val_set), 'config', json.dumps(cfg))

    net = Adapter(cfg['bands']).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg['lr_step_size'])
    km = KMeansCacheScheduler(cfg['kmeans_cache_update']); reset_cache(len(train_set))
    crit = torch.nn.L1Loss(reduction='mean')
    history, best, start = [], dict(val_l1=float('inf'), epoch=None), 0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        if (ck['sensor'], ck['seed']) != (a.sensor, a.seed): raise SystemExit(f"resume mismatch: {ck['sensor']} s{ck['seed']}")
        net.net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        km.current_epoch = ck['epoch']; history, best, start = ck['history'], ck['best'], ck['epoch']
        set_rng_state(ck['rng']); say('resumed from', a.resume, 'epoch', start)
    else:
        torch.save(dict(model=net.net.state_dict(), epoch=0, sensor=a.sensor, seed=a.seed), out / 'initial.pt')
    params = sum(q.numel() for q in net.parameters())
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, drop_last=False, pin_memory=True)
    durations = []

    for epoch in range(start + 1, a.epochs + 1):
        t0 = time.time(); km.step()
        if a.resume and epoch == start + 1: kmeans_cache.cache_mode = 'init'  # cache was not stored: rebuild it once
        mode = kmeans_cache.cache_mode; net.train(); losses = []
        for b in train_loader:
            opt.zero_grad()
            sr = net(b['pan'].to(dev), b['lms'].to(dev), index=b['index'].to(dev))
            loss = crit(sr, b['gt'].to(dev)); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        row = dict(epoch=epoch, train_l1=float(np.mean(losses)), lr=opt.param_groups[0]['lr'], kmeans_cache=mode, seconds=None)
        if epoch % cfg['val_interval'] == 0 or epoch == a.epochs:
            row['val_l1'], row['val_mse'] = validate(net, val_loader, dev)
            if row['val_l1'] < best['val_l1']:
                best = dict(val_l1=row['val_l1'], val_mse=row['val_mse'], epoch=epoch)
                torch.save(dict(model=net.net.state_dict(), epoch=epoch, val_l1=row['val_l1'], val_mse=row['val_mse'],
                                sensor=a.sensor, seed=a.seed, parameters=params, selection='lowest validation L1, every 10 epochs'),
                           out / 'best.pt')
        row['seconds'] = time.time() - t0; history.append(row); durations.append(row['seconds'])
        torch.save(dict(model=net.net.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch,
                        history=history, best=best, rng=rng_state(), sensor=a.sensor, seed=a.seed), out / 'latest.tmp')
        (out / 'latest.tmp').replace(out / 'latest.pt')
        (out / 'history.json').write_text(json.dumps(dict(sensor=a.sensor, seed=a.seed, parameters=params, recipe=cfg, best=best,
                                                          history=history), indent=1) + '\n', encoding='utf-8')
        say(f"epoch {epoch}/{a.epochs} train_l1 {row['train_l1']:.6f} cache {mode}"
            + (f" val_l1 {row['val_l1']:.6f} val_mse {row['val_mse']:.8f}" if 'val_l1' in row else '') + f" {row['seconds']:.0f}s best {best['epoch']}")
        if a.deadline and epoch < a.epochs and time.time() + 1.5 * max(durations[-12:]) > a.deadline:
            say('deadline: stopping after epoch', epoch, '(resume with --resume latest.pt)'); break
    else:
        say('finished', a.epochs, 'epochs; best epoch', best['epoch'], 'val_l1', best['val_l1'])


if __name__ == '__main__':
    main()
