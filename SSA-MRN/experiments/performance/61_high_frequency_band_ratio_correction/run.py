"""61 phase 1 Windows controller: caches, precomputed C2MS outputs, preflight, sequential queue (train then evaluate each run).

prepare: hard-links the verified normalized and 2x coarser caches of cache_reuse_source (SHA-256 checked against its
data_manifest) and writes the FP32 C2MS outputs of every train/val patch for each control seed (y_*.npy).
preflight: zero-initialized heads reproduce the C2MS output, the cached C2MS output equals a fresh forward, finite losses.
run: the queue of config.json phase1.queue; H3 takes the lambda of the validation-selected H2 run (config rule).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
from train import ROOT, KST, now, sha, write, run_name, c2ms_model, cache, setup  # noqa: E402
import model as M  # noqa: E402


def source_hashes():
    perf = HERE.parent; e54 = perf / '54_multi_sensor_scale_mixed_training'; e43 = perf / '43_normalized_parallel_capacity'
    paths = [HERE / f for f in ('model.py', 'train.py', 'run.py', 'config.json')] + [e54 / 'model.py', e54 / 'prepare.py', e43 / 'model.py',
             *sorted((e43 / 'provenance/source').rglob('*.py')), *(ROOT / 'common' / f for f in ('interpolation.py', 'metrics.py', 'observation.py', 'evaluation_protocol.py'))]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def fingerprint(cfg): return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()


def phase1_config():
    return dict(json.loads((HERE / 'config.json').read_text(encoding='utf8'))['phase1'])


def prepare():
    cfg = phase1_config(); root = Path(cfg['runtime']); root.mkdir(parents=True, exist_ok=True); setup(42)  # TF32 off for the cached C2MS outputs
    if (root / 'runtime_config.json').exists(): raise RuntimeError('runtime already prepared; do not overwrite')
    if shutil.disk_usage(root).free < 10 * 1024 ** 3: raise RuntimeError('less than 10 GiB free')
    source = Path(cfg['cache_reuse_source']); old = json.loads((source / 'data_manifest.json').read_text(encoding='utf8')); records = {}
    for sensor in cfg['sensors']:
        records[sensor] = {}; d = root / 'cache' / sensor; d.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'val'):
            rec = old[sensor][split]
            if rec['sha256'] != cfg['data_sha256'][sensor][split]: raise ValueError('data sha mismatch ' + sensor + split)
            records[sensor][split] = rec
            for coarse in (False, True):
                for k in ('pan', 'lms', 'ms', 'gt'):
                    name = f"{'coarse_' if coarse else ''}{split}_{k}.npy"; src = source / 'cache' / sensor / name; dest = d / name
                    expected = rec['coarse'][k]['sha256'] if coarse else rec['arrays'][k]['normalized_cache_sha256']
                    if not dest.exists(): os.link(src, dest)
                    if sha(dest) != expected: raise ValueError('cache mismatch ' + str(dest))
            print('VERIFIED CACHE', sensor, split, flush=True)
            # FP32 C2MS outputs of every patch (original and coarse) for each control seed
            for seed in cfg['seeds']:
                net = c2ms_model(cfg, sensor, seed); rec.setdefault('y', {})
                for coarse in (False, True):
                    pre = 'coarse_' if coarse else ''; src = {k: np.load(d / f'{pre}{split}_{k}.npy', mmap_mode='r') for k in ('pan', 'lms', 'ms')}
                    tmp = d / f'{pre}{split}_y_s{seed}_building.npy'
                    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=src['lms'].shape)
                    with torch.no_grad():
                        for i in range(0, len(arr), 128):
                            y = net(*[torch.from_numpy(np.array(src[k][i:i + 128])).cuda() for k in ('pan', 'lms', 'ms')]).float().cpu().numpy()
                            if not np.isfinite(y).all(): raise FloatingPointError('nonfinite C2MS output')
                            arr[i:i + len(y)] = y
                    arr.flush(); del arr; dest = d / f'{pre}{split}_y_s{seed}.npy'; os.replace(tmp, dest)
                    rec['y'][f'{pre}s{seed}'] = sha(dest); print('C2MS OUTPUT', sensor, split, pre or 'original', seed, flush=True)
    cfg['source_hashes'] = source_hashes(); cfg['created_kst'] = now()
    write(root / 'data_manifest.json', records); cfg['cache_manifest_sha256'] = sha(root / 'data_manifest.json'); cfg['fingerprint'] = fingerprint(cfg)
    write(root / 'runtime_config.json', cfg)
    write(root / 'environment.json', dict(time_kst=now(), python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                          free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS', flush=True)


def preflight():
    cfg = json.loads((Path(phase1_config()['runtime']) / 'runtime_config.json').read_text(encoding='utf8')); setup(42); checks = {}
    for sensor in cfg['sensors']:
        seed = cfg['seeds'][0]; c = cache(cfg, sensor, seed, 'val'); cc = cache(cfg, sensor, seed, 'val', True); net = c2ms_model(cfg, sensor, seed)
        idx = np.arange(0, len(c['y']), 97)[:16]; b = {k: torch.from_numpy(np.array(c[k][idx])).cuda() for k in c}
        ms = torch.from_numpy(np.array(np.load(Path(cfg['runtime']) / 'cache' / sensor / 'val_ms.npy', mmap_mode='r')[idx])).cuda()
        with torch.no_grad(): fresh = net(b['pan'], b['lms'], ms)
        row = dict(cached_vs_fresh_c2ms_max_abs=float((fresh - b['y']).abs().max()))
        for arm in M.ARMS:
            head = M.Head(arm, sensor, seed).cuda()
            with torch.no_grad(): p = head(b['y'], b['pan'], b['lms']); pc = head(*[torch.from_numpy(np.array(cc[k][idx])).cuda() for k in ('y', 'pan', 'lms')])
            row[f'{arm}_initial_vs_c2ms_max_abs'] = float((p - b['y']).abs().max())
            row[f'{arm}_coarse_initial_vs_c2ms_max_abs'] = float((pc - torch.from_numpy(np.array(cc['y'][idx])).cuda()).abs().max())
            row[f'{arm}_parameters'] = sum(x.numel() for x in head.parameters())
            head.train(); loss = (p - b['gt']).square().mean() + 0.01 * M.angle(head(b['y'], b['pan'], b['lms']), b['gt']); loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(x.grad).all() for x in head.parameters() if x.grad is not None): raise FloatingPointError('preflight ' + arm)
        if row['cached_vs_fresh_c2ms_max_abs'] > 1e-5 or max(v for k, v in row.items() if 'initial_vs' in k) > 1e-5: raise AssertionError(json.dumps(row))
        checks[sensor] = row
    write(Path(cfg['runtime']) / 'preflight.json', dict(time_kst=now(), checks=checks)); print('PREFLIGHT PASS', json.dumps(checks), flush=True)


def h3_lambda(cfg, root, sensor, seed):
    """Lambda of the H2 run whose selected epoch has the lowest validation SAM (config phase1.h3_lambda_rule)."""
    rows = []
    for lam in cfg['lambda_grid']:
        done = json.loads((root / 'runs' / run_name(sensor, 'H2', lam, seed) / 'training_complete.json').read_text(encoding='utf8'))
        rows.append((done['selected']['val_sam'], lam))
    return min(rows)[1]


def run(resume=False):
    root = Path(phase1_config()['runtime']); cp = root / 'runtime_config.json'; cfg = json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg) != cfg['fingerprint'] or source_hashes() != cfg['source_hashes']: raise RuntimeError('sealed config/source changed')
    if not (root / 'preflight.json').exists(): raise RuntimeError('preflight missing')
    write(root / 'controller_started.json', dict(time_kst=now(), pid=os.getpid(), fingerprint=cfg['fingerprint']))
    def job(phase, sensor, arm, lam, seed):
        out = root / 'runs' / run_name(sensor, arm, lam, seed); out.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, '-u', str(HERE / 'train.py'), '--config', str(cp), '--mode', phase, '--sensor', sensor, '--arm', arm, '--lam', str(lam), '--seed', str(seed)]
        if phase == 'train' and resume and (out / 'checkpoints/latest.pt').exists(): args.append('--resume')
        with (out / f'{phase}.log').open('a', encoding='utf8') as log: code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
        print(json.dumps(dict(time_kst=now(), phase=phase, run=out.name, returncode=code)), flush=True)
        if code: write(root / 'controller_failure.json', dict(time_kst=now(), phase=phase, run=out.name, returncode=code)); raise RuntimeError('job failed ' + out.name)
    for sensor, arm, lam, seed in cfg['queue']:
        if lam == 'h3_rule': lam = h3_lambda(cfg, root, sensor, seed); write(root / 'runs' / f'{sensor}_H3_lambda_s{seed}.json', dict(time_kst=now(), lam=lam, rule=cfg['h3_lambda_rule']))
        out = root / 'runs' / run_name(sensor, arm, lam, seed)
        if not (out / 'training_complete.json').exists(): job('train', sensor, arm, lam, seed)
        if not (out / 'evaluation_complete.json').exists(): job('evaluate', sensor, arm, lam, seed)
    write(root / 'all_complete.json', dict(time_kst=now(), fingerprint=cfg['fingerprint'], runs=len(cfg['queue'])))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['prepare', 'preflight', 'run'], required=True); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    {'prepare': prepare, 'preflight': preflight}.get(a.mode, lambda: run(a.resume))()
