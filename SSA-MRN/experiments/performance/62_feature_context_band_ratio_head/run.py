"""62 Windows controller: hard-linked caches, preflight, sequential queue (train then evaluate each arm).

prepare: hard-links the verified normalized and 2x coarser caches of cache_reuse_source (SHA-256 checked against its
data_manifest). preflight: every zero-initialized arm reproduces the C2MS output on validation patches (original and
coarse), head sizes are within max_head_parameters, gradients are finite, and a short step benchmark gives the ETA.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
from train import ROOT, now, sha, write, run_name, c2ms_model, cache, batch, setup  # noqa: E402
import model as M  # noqa: E402


def source_hashes():
    perf = HERE.parent; e54 = perf / '54_multi_sensor_scale_mixed_training'; e43 = perf / '43_normalized_parallel_capacity'
    paths = [HERE / f for f in ('model.py', 'train.py', 'run.py', 'config.json')] + [perf / '61_high_frequency_band_ratio_correction/model.py', e54 / 'model.py', e54 / 'prepare.py',
             e43 / 'model.py', *sorted((e43 / 'provenance/source').rglob('*.py')), *(ROOT / 'common' / f for f in ('interpolation.py', 'metrics.py', 'observation.py', 'evaluation_protocol.py'))]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def fingerprint(cfg): return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
def phase1_config(): return dict(json.loads((HERE / 'config.json').read_text(encoding='utf8'))['phase1'])
def runtime_config(): return json.loads((Path(phase1_config()['runtime']) / 'runtime_config.json').read_text(encoding='utf8'))


def prepare():
    cfg = phase1_config(); root = Path(cfg['runtime']); root.mkdir(parents=True, exist_ok=True)
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
    cfg['source_hashes'] = source_hashes(); cfg['created_kst'] = now()
    write(root / 'data_manifest.json', records); cfg['cache_manifest_sha256'] = sha(root / 'data_manifest.json'); cfg['fingerprint'] = fingerprint(cfg)
    write(root / 'runtime_config.json', cfg)
    write(root / 'environment.json', dict(time_kst=now(), python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                          free_bytes=shutil.disk_usage(root).free))
    print('PREPARE PASS', flush=True)


def preflight():
    cfg = runtime_config(); setup(42); checks = {}
    for sensor in cfg['sensors']:
        seed = cfg['seeds'][0]; c2ms = c2ms_model(cfg, sensor, seed); va, vac, tr, trc = cache(cfg, sensor, 'val'), cache(cfg, sensor, 'val', True), cache(cfg, sensor, 'train'), cache(cfg, sensor, 'train', True)
        idx = np.arange(0, len(va['pan']), 97)[:16]; b, bc = batch(va, idx), batch(vac, idx); row = {}
        with torch.no_grad(): y, f = c2ms(*b[:3]); yc, fc = c2ms(*bc[:3])
        row['feature_shape'] = list(f.shape)
        for arm in M.ARMS:
            head = M.Head(arm, sensor, seed).cuda()
            with torch.no_grad(): p = head(y, b[0], b[1], f); pc = head(yc, bc[0], bc[1], fc)
            row[arm] = dict(parameters=sum(x.numel() for x in head.parameters()), initial_vs_c2ms_max_abs=float((p - y).abs().max()), coarse_initial_vs_c2ms_max_abs=float((pc - yc).abs().max()))
            if row[arm]['parameters'] > cfg['max_head_parameters'] or max(row[arm]['initial_vs_c2ms_max_abs'], row[arm]['coarse_initial_vs_c2ms_max_abs']) > 1e-5: raise AssertionError(json.dumps(row))
            opt = torch.optim.Adam(head.parameters(), lr=cfg['lr_start']); head.train(); times = []
            for s in range(25):
                torch.cuda.synchronize(); t = time.perf_counter(); i = np.sort(np.random.default_rng(s).choice(len(tr['pan']), cfg['batch'], replace=False))
                bb, bbc = batch(tr, i), batch(trc, i)
                with torch.no_grad(): yy, ff = c2ms(*bb[:3]); yyc, ffc = c2ms(*bbc[:3])
                loss = (head(yy, bb[0], bb[1], ff) - bb[3]).square().mean() + (head(yyc, bbc[0], bbc[1], ffc) - bbc[3]).square().mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(x.grad).all() for x in head.parameters() if x.grad is not None): raise FloatingPointError('preflight ' + arm)
                opt.step(); torch.cuda.synchronize()
                if s >= 5: times.append(time.perf_counter() - t)
            steps = -(-len(tr['pan']) // cfg['batch'])
            row[arm].update(median_step_seconds=float(np.median(times)), train_epoch_seconds_estimate=float(np.median(times)) * steps, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
        checks[sensor] = row
    write(Path(cfg['runtime']) / 'preflight.json', dict(time_kst=now(), checks=checks)); print('PREFLIGHT PASS', json.dumps(checks), flush=True)


def run(resume=False):
    root = Path(phase1_config()['runtime']); cp = root / 'runtime_config.json'; cfg = json.loads(cp.read_text(encoding='utf8'))
    if fingerprint(cfg) != cfg['fingerprint'] or source_hashes() != cfg['source_hashes']: raise RuntimeError('sealed config/source changed')
    if not (root / 'preflight.json').exists(): raise RuntimeError('preflight missing')
    write(root / 'controller_started.json', dict(time_kst=now(), pid=os.getpid(), fingerprint=cfg['fingerprint']))
    def job(phase, sensor, arm, seed):
        out = root / 'runs' / run_name(sensor, arm, seed); out.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, '-u', str(HERE / 'train.py'), '--config', str(cp), '--mode', phase, '--sensor', sensor, '--arm', arm, '--seed', str(seed)]
        if phase == 'train' and resume and (out / 'checkpoints/latest.pt').exists(): args.append('--resume')
        with (out / f'{phase}.log').open('a', encoding='utf8') as log: code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
        print(json.dumps(dict(time_kst=now(), phase=phase, run=out.name, returncode=code)), flush=True)
        if code: write(root / 'controller_failure.json', dict(time_kst=now(), phase=phase, run=out.name, returncode=code)); raise RuntimeError('job failed ' + out.name)
    for sensor, arm, seed in cfg['queue']:
        out = root / 'runs' / run_name(sensor, arm, seed)
        if not (out / 'training_complete.json').exists(): job('train', sensor, arm, seed)
        if not (out / 'evaluation_complete.json').exists(): job('evaluate', sensor, arm, seed)
    write(root / 'all_complete.json', dict(time_kst=now(), fingerprint=cfg['fingerprint'], runs=len(cfg['queue'])))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['prepare', 'preflight', 'run'], required=True); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    {'prepare': prepare, 'preflight': preflight}.get(a.mode, lambda: run(a.resume))()
