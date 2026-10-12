"""Linux A6000 queue: staged repeatability, location ablation and sensor generalization, no automatic test selection."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'scripts/a6000_followup_plan.json'
RUN = ROOT / 'experiments/a6000_followup_123'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def inventory(plan):
    files = list((ROOT / 'src').rglob('*.py'))
    files += [Path(__file__), ROOT / 'scripts/train.py', ROOT / 'scripts/train_a6000_followup.py', ROOT / 'references/upstream/network.py']
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    data = {}
    for paths in plan['data'].values():
        for filename in paths.values():
            p = Path(filename).expanduser()
            s = p.stat()
            data[str(p.resolve())] = {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}
    references = {}
    for row in plan.get('references',[]):
        d=Path(row['path'])
        for filename in ('best.pt','latest.pt','complete.json','history.jsonl'):
            p=d/filename
            references[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'source': source, 'data': data, 'references': references}


def quota_free():
    result = subprocess.run(['quota', '-w'], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0].startswith('/dev/'):
            used, hard = int(fields[1].rstrip('*')), int(fields[3])
            if hard:
                return (hard - used) * 1024
    raise RuntimeError('Cannot read account quota; refusing to rely on filesystem free space')


def check(plan):
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from ssamrn.data.pancollection import PanCollectionH5
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if plan['parallel'] not in (1, 2, 3, 4):
        raise RuntimeError('Parallel jobs must be between one and four')
    if quota_free() < 512 * 1024**2:
        raise RuntimeError('At least 512 MiB of account quota must remain for checkpoints')
    missing = [str(Path(p).expanduser()) for paths in plan['data'].values() for p in paths.values() if not Path(p).expanduser().is_file()]
    if missing:
        raise RuntimeError('Upload pending: ' + ', '.join(missing))
    for sensor, paths in plan['data'].items():
        for split, path in paths.items():
            if Path(path).expanduser().stat().st_size != plan['expected_bytes'][sensor][split]:
                raise RuntimeError('Upload incomplete or unexpected source file: ' + path)
    shapes = {}
    for sensor, paths in plan['data'].items():
        train, val = (PanCollectionH5(Path(paths[k]).expanduser(), sensor) for k in ('train', 'val'))
        if train.path.resolve() == val.path.resolve() or train.channels != val.channels:
            raise RuntimeError('Invalid train/validation split')
        if train.channels != (8 if sensor == 'WV3' else 4):
            raise RuntimeError('Sensor band count mismatch')
        for dataset in (train, val):
            dataset[0]; dataset[len(dataset)-1]; dataset.close()
        shapes[sensor] = {'train': len(train), 'validation': len(val)}
    if [r['stage'] for r in plan['runs']]!=sorted(r['stage'] for r in plan['runs']):
        raise RuntimeError('Stages must be ordered')
    for row in plan.get('references',[]):
        original=json.loads((Path(row['path'])/'complete.json').read_text())
        if original['config']!=config(plan,row) or original['epoch']!=plan['training']['epochs']:
            raise RuntimeError('Reference conditions differ')
        snapshot=json.loads((Path(row['path']).parent/'inventory.json').read_text())['source']
        for filename in ['src/ssamrn/models/ssa_mrn.py','src/ssamrn/models/interp23.py','scripts/train.py']:
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=snapshot[filename]:
                raise RuntimeError('Shared reference source differs: '+filename)
    return {'status': 'ready', 'runs': len(plan['runs']), 'parallel': plan['parallel'], 'data': shapes, 'quota_free_bytes': quota_free()}


def config(plan, row):
    paths = plan['data'][row['sensor']]
    condition={k:v for k,v in row.items() if k not in ('stage','path')}
    return dict(plan['training'], **condition, train=str(Path(paths['train']).expanduser()), val=str(Path(paths['val']).expanduser()), loss='baseline', weight=0.)


def name(row):
    return f"{row['sensor']}_{row['variant']}_k{row['k']}_s{row['seed']}"


def worker(plan):
    lock = (RUN / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    try:
        expected = json.loads((RUN / 'inventory.json').read_text())
        pending = list(plan['runs']);stage=1
        write(RUN/'references.json',plan.get('references',[]))
        while pending or active:
            if inventory(plan) != expected:
                raise RuntimeError('Source or data changed; queue halted')
            for label, (child, log) in list(active.items()):
                code = child.poll()
                if code is not None:
                    log.close()
                    del active[label]
                    if code:
                        raise RuntimeError(f'{label} failed; see its train.log')
            if not active and pending:stage=pending[0]['stage']
            while pending and pending[0]['stage']==stage and len(active) < plan['parallel']:
                if quota_free() < 256 * 1024**2:
                    raise RuntimeError('Checkpoint quota reserve exhausted')
                row = pending.pop(0)
                label = name(row)
                directory = RUN / label
                c = config(plan, row)
                directory.mkdir(exist_ok=True)
                cfg = directory / 'config.json'
                if cfg.exists() and json.loads(cfg.read_text()) != c:
                    raise RuntimeError('Existing config mismatch')
                write(cfg, c)
                done = directory / 'complete.json'
                if done.exists():
                    if json.loads(done.read_text())['config'] != c:
                        raise RuntimeError('Completed config mismatch')
                    continue
                command = [sys.executable, str(ROOT / 'scripts/train_a6000_followup.py'), '--config', str(cfg), '--output', str(directory)]
                if (directory / 'latest.pt').exists():
                    command += ['--resume', 'latest']
                log = (directory / 'train.log').open('a')
                env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[label] = (child, log)
            write(RUN / 'state.json', {'pid': os.getpid(), 'status': 'running', 'active': {k: v[0].pid for k, v in active.items()}, 'pending': len(pending), 'stage': stage, 'heartbeat': time.time()})
            if active:
                time.sleep(5)
        scores = {name(row): json.loads((RUN / name(row) / 'complete.json').read_text())['best_mse'] for row in plan['runs']}
        for row in plan.get('references',[]):
            scores[name(row)]=json.loads((Path(row['path'])/'complete.json').read_text())['best_mse']
        write(RUN / 'validation_scores.json', scores)
        write(RUN / 'state.json', {'pid': os.getpid(), 'status': 'finished', 'active': {}, 'heartbeat': time.time()})
    except BaseException as e:
        # Stop only children created by this controller; retain epoch checkpoints.
        for child, log in active.values():
            if child.poll() is None:
                child.terminate()
        for child, log in active.values():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.close()
        write(RUN / 'state.json', {'pid': os.getpid(), 'status': 'failed', 'active': {}, 'error': str(e), 'heartbeat': time.time()})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['check', 'start', 'resume-latest', 'status', 'logs', 'worker'])
    a = p.parse_args()
    plan = json.loads(PLAN.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    statefile = RUN / 'state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {'status': 'not_started'}
    if a.command == 'check':
        print(json.dumps(check(plan), indent=2))
    elif a.command in ('start', 'resume-latest'):
        if alive(state.get('pid')) or any(alive(pid) for pid in state.get('active', {}).values()):
            raise RuntimeError('Queue or its training process is already running')
        if a.command == 'start' and any(RUN.glob('*/latest.pt')):
            raise RuntimeError('Use resume-latest for saved checkpoints')
        check(plan)
        gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if gpu_pids:
            raise RuntimeError('GPU already in use; inspect other jobs before starting')
        for filename, value in [('plan.json', plan), ('inventory.json', inventory(plan))]:
            path = RUN / filename
            if path.exists() and json.loads(path.read_text()) != value:
                raise RuntimeError('Plan/source/data mismatch; use a new experiment directory')
            write(path, value)
        with (RUN / 'controller.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(statefile, {'pid': child.pid, 'status': 'starting', 'active': {}, 'heartbeat': time.time()})
        print('controller PID=', child.pid)
    elif a.command == 'worker':
        worker(plan)
    elif a.command == 'status':
        print(json.dumps(dict(state, controller_alive=alive(state.get('pid'))), indent=2))
    else:
        for path in sorted(RUN.glob('*/train.log')):
            print(path.parent.name)
            print('\n'.join(path.read_text().splitlines()[-3:]))


if __name__ == '__main__':
    main()
