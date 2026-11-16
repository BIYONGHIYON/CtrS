# Kaggle notebook for the CANNet benchmark (from version 2, 2026-11-16: single seed 42), percent format: each "# %%" is one notebook cell.
# Import into Kaggle with File > Import Notebook. Inputs: dataset ctrs-pan-sharpening-dataset; accelerator GPU T4 x2; internet on.

# %% [markdown]
# # CANNet benchmark (CtrS, SSA-MRN/benchmarks/cannet)
#
# Official code Duanyll/CANConv @ `0b7cc6f4ed257fcdca806d1d4fe8ec2ebf4beee9` (unmodified), our PanCollection split from the private dataset `ctrs-pan-sharpening-dataset`,
# evaluation with `SSA-MRN/common/benchmark_evaluate.py` (CtrS main). Accelerator: **GPU T4 x2**, Internet **on**.
#
# 1. build the native operators (pwacnative, libKMCUDA) for this GPU
# 2. `author`: evaluate the published weights (QB, GF2, WV3 and WV2 = WV3 weights, reference)
# 3. `retrain`: official recipe, seed 42 on QB (single seed, user decision 2026-11-16; version 1 also trained seed 43, not continued), resumable; stops before the 12 h limit.
#    To continue in the next version, add this notebook's previous output as an input: `latest.pt` is picked up automatically.

# %%
import os, sys, time, json, glob, shutil, subprocess
from pathlib import Path
T0 = time.time()
TRAIN_HOURS = 10.8          # training stops before this; the rest is left for evaluation and saving the output
JOBS = [('QB', 42)]   # single seed (user decision 2026-11-16); version 1 ran [('QB', 42), ('QB', 43), ('QB', 44)]
W = Path('/kaggle/working'); TMP = Path('/tmp/ctrs'); TMP.mkdir(exist_ok=True)
def sh(cmd, cwd=None):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if r.returncode: raise RuntimeError(f'failed ({r.returncode}): {cmd}')
    return r.stdout
sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())

# %%
# 1. official code + native operators
UP = TMP / 'CANConv'
if not (UP / 'canconv/layers/pwac/pwacnative.so').exists():
    if not UP.exists():
        sh(f'git clone -q https://github.com/Duanyll/CANConv.git {UP}')
        sh('git checkout -q 0b7cc6f4ed257fcdca806d1d4fe8ec2ebf4beee9 && git submodule update --init --recursive -q', cwd=UP)
    sh('pip install -q ninja cmake einops')
    cc = torch.cuda.get_device_capability(0); arch = f'{cc[0]}{cc[1]}'
    py = sys.executable
    sh(f'cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH={arch} -DCMAKE_CUDA_ARCHITECTURES={arch} '
       f'-DPython3_EXECUTABLE={py} -DPYTHON_EXECUTABLE={py} -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -Bbuild -G Ninja', cwd=UP)
    sh('cmake --build build --config Release --target all', cwd=UP)
print(sh('git rev-parse HEAD && git -C kmcuda rev-parse HEAD', cwd=UP))
print(sorted(p.name for p in (UP / 'canconv/layers').rglob('*.so')))

# CtrS evaluation code (public main) and the adapter folder
CTRS = TMP / 'CtrS'
if not CTRS.exists():  # sparse: only the evaluation code and the C2MS-SP reference numbers (the full repository is ~5 GB)
    sh(f'git clone -q --depth 1 --filter=blob:none --sparse https://github.com/BIYONGHIYON/CtrS.git {CTRS}')
    sh('git sparse-checkout set SSA-MRN/common SSA-MRN/docs/images/c2ms_sp', cwd=CTRS)
print('CtrS', sh('git rev-parse HEAD', cwd=CTRS))
BENCH = W / 'cannet'; BENCH.mkdir(exist_ok=True)
if not (BENCH / 'upstream').exists(): os.symlink(UP, BENCH / 'upstream')

# %%
%%writefile /kaggle/working/cannet/model.py
"""CANNet adapter: official code in upstream/ (Duanyll/CANConv, unmodified); inputs and output in DN / scale.

The official network takes (pan, lms) and returns lms + residual; ms is not used. In eval mode the k-means layers ignore the
training cache and cluster every image from scratch, as in the official test script. The native operators (pwacnative, libKMCUDA)
must be built in upstream/ first (upstream/build.sh, CUDA only).
"""
import sys
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / 'upstream'
sys.path.insert(0, str(UPSTREAM))
from canconv.models.cannet.model import CANNet  # noqa: E402  (official code, unmodified)

CHANNELS, CLUSTERS, THRESHOLD = 32, 32, 0.005  # canconv/models/cannet/default.json


class Adapter(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.net = CANNet(bands, CHANNELS, CLUSTERS, THRESHOLD)

    def forward(self, pan, lms, ms=None, index=None):
        return self.net(pan, lms, index)  # official input order; index selects the k-means cache during training only


def build(bands, sensor, setting, seed):
    return Adapter(bands)


def load(net, path):
    """Author weights are a bare CANNet state_dict; our checkpoints store it under 'model'."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state, strict=True)

# %%
%%writefile /kaggle/working/cannet/train.py
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

# %%
# data: Kaggle flattens folder names into file names, so link the PanCollection files under their own names
DATA = TMP / 'data'; DATA.mkdir(exist_ok=True)
names = [f'{k}_{s}.h5' for s in ('qb', 'gf2', 'wv3') for k in ('train', 'valid')] + \
        [f'test_{s}{o}_multiExm1.h5' for s in ('qb', 'gf2', 'wv3', 'wv2') for o in ('', '_OrigScale')]
files = [p for p in Path('/kaggle/input').rglob('*.h5') if p.stat().st_size > 1 << 20]
for n in names:
    hits = sorted(p for p in files if (p.name == n or p.name.endswith('__' + n)) and not p.name.endswith('._' + n))
    print(n, '->', [h.name for h in hits][:3])
    if hits and not (DATA / n).exists(): os.symlink(hits[0], DATA / n)

# %%
# 2. author weights (published, no training); WV2 uses the WV3 weights (transfer, reference)
# one visible GPU: kmcuda otherwise spreads k-means over all GPUs and needs peer access, which the two T4s do not have
EVAL = f'CUDA_VISIBLE_DEVICES=0 {sys.executable} {CTRS}/SSA-MRN/common/benchmark_evaluate.py --folder {BENCH} --data {DATA} --device cuda'
for sensor, w in (('QB', 'qb'), ('GF2', 'gf2'), ('WV3', 'wv3'), ('WV2', 'wv3')):
    if (BENCH / f'results/author_{sensor}/metrics.json').exists(): continue
    try: sh(f'{EVAL} --setting author --sensor {sensor} --checkpoint {UP}/weights/cannet_{w}.pth')
    except Exception as e: print('author eval failed', sensor, e)

# %%
# 3. retrain: resume from a previous version's output if attached as input
RUNS = W / 'runs'; RUNS.mkdir(exist_ok=True)
for prev in Path('/kaggle/input').rglob('retrain_*_s*/latest.pt'):
    dst = RUNS / prev.parent.name
    old = json.loads((dst / 'history.json').read_text())['history'][-1]['epoch'] if (dst / 'history.json').exists() else -1
    new = json.loads((prev.parent / 'history.json').read_text())['history'][-1]['epoch']
    if new > old: shutil.copytree(prev.parent, dst, dirs_exist_ok=True); print('restored', dst.name, 'epoch', new)
for prev in Path('/kaggle/input').rglob('cannet/results/*/metrics.json'):
    dst = BENCH / 'results' / prev.parent.name / 'metrics.json'
    if not dst.exists(): dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(prev, dst)

def epochs_done(sensor, seed):
    h = RUNS / f'retrain_{sensor}_s{seed}/history.json'
    return json.loads(h.read_text())['history'][-1]['epoch'] if h.exists() else 0

DEADLINE = T0 + TRAIN_HOURS * 3600
queue = [j for j in JOBS if epochs_done(*j) < 500]
running = {}
print('queue', queue, 'deadline in', round((DEADLINE - time.time()) / 3600, 2), 'h')
while queue or running:
    for gpu in range(torch.cuda.device_count()):
        if gpu not in running and queue and time.time() < DEADLINE - 1800:
            sensor, seed = queue.pop(0); latest = RUNS / f'retrain_{sensor}_s{seed}/latest.pt'
            cmd = [sys.executable, str(BENCH / 'train.py'), '--sensor', sensor, '--seed', str(seed), '--data', str(DATA),
                   '--out', str(RUNS), '--deadline', str(DEADLINE)] + (['--resume', str(latest)] if latest.exists() else [])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            logf = open(RUNS / f'stdout_{sensor}_s{seed}.log', 'a')
            running[gpu] = (sensor, seed, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)); print('start', gpu, sensor, seed, flush=True)
    time.sleep(300)
    for gpu, (sensor, seed, proc) in list(running.items()):
        log = RUNS / f'retrain_{sensor}_s{seed}/train.log'
        if log.exists(): print(f'[{(time.time()-T0)/3600:.2f}h gpu{gpu}]', log.read_text().strip().splitlines()[-1], flush=True)
        if proc.poll() is not None:
            print('exit', sensor, seed, proc.returncode, 'epochs', epochs_done(sensor, seed), flush=True); del running[gpu]
            if proc.returncode: print(open(RUNS / f'stdout_{sensor}_s{seed}.log').read()[-3000:])
    if time.time() > DEADLINE and not running: break

# %%
# 4. evaluate finished runs (all 500 epochs) on QB RR/FR test; best.pt goes to checkpoints/
for sensor, seed in JOBS:
    run = RUNS / f'retrain_{sensor}_s{seed}'
    if epochs_done(sensor, seed) < 500 or (BENCH / f'results/retrain_{sensor}_s{seed}/metrics.json').exists(): continue
    ck = BENCH / f'checkpoints/retrain_{sensor}_s{seed}'; ck.mkdir(parents=True, exist_ok=True); shutil.copy(run / 'best.pt', ck / 'best.pt')
    sh(f'{EVAL} --setting retrain --sensor {sensor} --seed {seed}')
    shutil.copy(run / 'history.json', BENCH / f'results/retrain_{sensor}_s{seed}/history.json')
sh(f'{sys.executable} {CTRS}/SSA-MRN/common/benchmark_evaluate.py --folder {BENCH} --summary')
for sensor, seed in JOBS:
    log = RUNS / f'retrain_{sensor}_s{seed}/train.log'
    print(sensor, seed, 'epochs', epochs_done(sensor, seed), log.read_text().strip().splitlines()[-1] if log.exists() else '')
print('elapsed', round((time.time() - T0) / 3600, 2), 'h')
