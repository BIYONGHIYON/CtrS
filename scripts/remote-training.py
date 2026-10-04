"""Windows LIB training controller. No third-party dependency for control commands."""
import argparse
import contextlib
import ctypes
from ctypes import wintypes
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'SSA-MRN/experiments/logs/remote-control'
RUNS = ROOT / 'SSA-MRN/experiments/checkpoints/remote-runs'
CONFIG = ROOT / 'SSA-MRN/configs/lib_rgb_hsi_triple17_k4_spectral_tiles.json'
PYTHON = Path('C:/CtrS/.venv/Scripts/python.exe')


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return default


def write(path, value):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


@contextlib.contextmanager
def lock(name):
    import msvcrt
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / name).open('a+b') as f:
        f.seek(0)
        if not f.read(1):
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another controller operation holds the lock; retry.')
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class Job:
    """Own only this run's process tree; kill it if the worker dies or closes the job."""
    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_longlong), ('JobTime', ctypes.c_longlong),
                        ('Flags', wintypes.DWORD), ('MinWS', ctypes.c_size_t),
                        ('MaxWS', ctypes.c_size_t), ('Limit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('Priority', wintypes.DWORD),
                        ('Scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in
                        ('ReadCount', 'WriteCount', 'OtherCount', 'ReadBytes', 'WriteBytes', 'OtherBytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('Basic', Basic), ('IO', IO), ('ProcessMem', ctypes.c_size_t),
                        ('JobMem', ctypes.c_size_t), ('PeakProcess', ctypes.c_size_t),
                        ('PeakJob', ctypes.c_size_t)]
        k = self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateJobObject.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.NtResumeProcess = ctypes.WinDLL('ntdll').NtResumeProcess
        k.NtResumeProcess.argtypes = [wintypes.HANDLE]
        k.NtResumeProcess.restype = ctypes.c_long
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def launch(self, command, log):
        # Suspended creation prevents Python/DataLoader children escaping before assignment.
        p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT,
                             creationflags=0x00000004 | 0x08000000)
        try:
            if not self.k.AssignProcessToJobObject(self.handle, int(p._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            status = self.k.NtResumeProcess(int(p._handle))
            if status < 0:
                raise RuntimeError(f'NtResumeProcess failed: {status}')
            return p
        except BaseException:
            p.kill()
            p.wait()
            raise

    def stop(self):
        if not self.k.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        self.wait_empty()

    def wait_empty(self):
        class Accounting(ctypes.Structure):
            _fields_ = [(n, ctypes.c_longlong) for n in ('User', 'Kernel', 'PeriodUser', 'PeriodKernel')] + [
                (n, wintypes.DWORD) for n in ('Faults', 'Total', 'Active', 'Terminated')]
        deadline = time.time() + 20
        while time.time() < deadline:
            info = Accounting()
            if not self.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.Active == 0:
                return
            time.sleep(0.1)
        raise RuntimeError('Process-tree termination timeout. Check status before any new start.')

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def legacy_processes():
    # Do not kill any of these. Refuse to compete with training outside this controller.
    script = r"""$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_lib\.py' } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"""
    p = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
                       capture_output=True, text=True, timeout=20)
    if p.returncode:
        raise RuntimeError('Cannot verify existing training processes: ' + p.stderr.strip())
    return json.loads(p.stdout) if p.stdout.strip() else []


def prepare(request):
    cfg = read(CONFIG)
    run = RUNS / request['id']
    run.mkdir(parents=True, exist_ok=False)
    log_dir = BASE / 'runs' / request['id']
    log_dir.mkdir(parents=True, exist_ok=False)
    # Snapshot config; never write original configuration or checkpoint directory.
    snapshot = log_dir / 'config.json'
    cfg['data_root'] = 'C:/CtrS/SSA-MRN/data/dataset/RGB-HSI/LIB-HSI'
    cfg['output_dir'] = str(run)
    write(snapshot, cfg)
    command = [str(PYTHON), '-u', str(ROOT / 'SSA-MRN/scripts/train_lib.py'),
               '--config', str(snapshot), '--output-dir', str(run)]
    source = request.get('checkpoint')
    if source:
        # Copy the resume snapshot into the NEW run; train never reads a changing source.
        source = Path(source)
        if not source.is_file():
            raise RuntimeError('Checkpoint does not exist: ' + str(source))
        seed = run / 'resume-source.pt'
        shutil.copy2(source, seed)
        command += ['--resume', str(seed)]
        # Retain inherited best only when it belongs to this resume history.
        import torch
        resumed = torch.load(seed, map_location='cpu', weights_only=True)
        prior_best = source.parent / 'best.pt'
        candidate = run / 'inherited-best.pt'
        if prior_best.is_file():
            shutil.copy2(prior_best, candidate)
            best = torch.load(candidate, map_location='cpu', weights_only=True)
            if (best.get('epoch', float('inf')) <= resumed['epoch'] and
                    best.get('best_mse') == resumed['best_mse'] and
                    best.get('config') == resumed.get('config')):
                candidate.replace(run / 'best.pt')
            # Otherwise keep the copy for provenance; do not call it this run's best.
        del resumed
    meta = {'id': request['id'], 'status': 'starting', 'output': str(run),
            'log': str(log_dir / 'train.log'), 'source': source and str(source),
            'mode': request['mode'], 'started_at': dt.datetime.now().isoformat(),
            'command': command}
    write(log_dir / 'run.json', meta)
    return meta


def worker():
    with lock('worker.lock'):
        (BASE / 'requests').mkdir(exist_ok=True)
        (BASE / 'acks').mkdir(exist_ok=True)
        state_path = BASE / 'state.json'
        meta = read(state_path, {})
        # Previous job is killed by Windows when its owning worker exits.
        if meta.get('status') in ('running', 'starting'):
            meta['status'] = 'interrupted'
            write(state_path, meta)
        process = job = log = None
        try:
            while True:
                write(BASE / 'heartbeat.json', {'time': time.time(), 'pid': os.getpid()})
                if process and process.poll() is not None:
                    # Clean up any DataLoader/launcher descendants after the main process exits.
                    job.stop()
                    meta.update(status='finished' if process.returncode == 0 else 'failed',
                                exit_code=process.returncode, ended_at=dt.datetime.now().isoformat())
                    job.close()
                    log.close()
                    process = job = log = None
                    write(state_path, meta)
                    write(Path(meta['log']).parent / 'run.json', meta)
                for file in sorted((BASE / 'requests').glob('*.json')):
                    request = read(file)
                    # Never replay commands left over after worker failure or client timeout.
                    ack = BASE / 'acks' / file.name
                    if ack.exists():
                        file.unlink()
                        continue
                    result = {'ok': False, 'id': request['id']}
                    try:
                        if time.time() - request['time'] > 30:
                            raise RuntimeError('Request expired; submit again after status check.')
                        if request['action'] == 'stop':
                            if not process or meta['id'] != request['run_id']:
                                raise RuntimeError('Requested run is no longer active; nothing was killed.')
                            job.stop()
                            process.wait(timeout=20)
                            meta.update(status='stopped', exit_code=process.returncode,
                                        ended_at=dt.datetime.now().isoformat())
                            job.close()
                            log.close()
                            process = job = log = None
                            write(state_path, meta)
                            write(Path(meta['log']).parent / 'run.json', meta)
                        else:
                            if process:
                                raise RuntimeError('A managed training run is already active.')
                            if legacy_processes():
                                raise RuntimeError('Existing train_lib.py process detected. No process was changed.')
                            meta = prepare(request)
                            write(state_path, meta)
                            job = Job()
                            log = open(meta['log'], 'ab', buffering=0)
                            process = job.launch(meta['command'], log)
                            meta.update(status='running', pid=process.pid)
                            write(state_path, meta)
                            write(Path(meta['log']).parent / 'run.json', meta)
                        result.update(ok=True, run=meta)
                    except Exception as e:
                        result['error'] = str(e)
                        if request['action'] != 'stop' and not process:
                            if job:
                                job.close()
                                job = None
                            if log:
                                log.close()
                                log = None
                            if meta.get('id') == request['id']:
                                meta.update(status='failed', error=str(e))
                                write(state_path, meta)
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                    write(ack, result)
                    file.unlink()
                time.sleep(1)
        finally:
            if job:
                job.close()
            if log:
                log.close()
            write(BASE / 'heartbeat.json', {'time': 0, 'pid': os.getpid()})


def online():
    hb = read(BASE / 'heartbeat.json', {})
    return time.time() - hb.get('time', 0) < 15


def submit(args):
    with lock('client.lock'):
        if not online():
            raise RuntimeError('Controller offline. Start CtrS-LIB-Control on server; no training was queued.')
        state = read(BASE / 'state.json', {})
        if args.command == 'stop':
            if state.get('status') != 'running':
                raise RuntimeError('No managed training is running; no process was changed.')
            request = {'action': 'stop', 'run_id': state['id']}
        else:
            if state.get('status') in ('running', 'starting'):
                raise RuntimeError('Training already active. Use status/logs/stop.')
            cfg = read(CONFIG)
            source = None
            if args.command.startswith('resume-'):
                directory = Path(args.from_dir) if args.from_dir else ROOT / cfg['output_dir']
                directory = directory.resolve()
                source = directory / ('latest.pt' if args.command == 'resume-latest' else 'best.pt')
                if not source.is_file():
                    raise RuntimeError('Checkpoint not found: ' + str(source))
            request = {'action': 'start', 'mode': args.command,
                       'checkpoint': source and str(source)}
        request.update(id=dt.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:12],
                       time=time.time())
        write(BASE / 'requests' / (request['id'] + '.json'), request)
        ack = BASE / 'acks' / (request['id'] + '.json')
        deadline = time.time() + 60
        while time.time() < deadline:
            result = read(ack)
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if not result['ok']:
                    raise RuntimeError(result['error'])
                return
            time.sleep(0.5)
        raise RuntimeError('Acknowledgement timeout. Do not resubmit blindly; inspect status and acknowledgements.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'start', 'resume-latest', 'resume-best',
                                          'stop', 'status', 'logs', 'inspect'])
    parser.add_argument('--from-dir', help='Read-only checkpoint directory for resume; default: original config output_dir')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This controller runs on the Windows server, including commands sent via Remote-SSH.')
    if args.command == 'worker':
        worker()
    elif args.command == 'inspect':
        import torch
        cfg = read(CONFIG)
        directory = Path(args.from_dir) if args.from_dir else ROOT / cfg['output_dir']
        for name in ('latest.pt', 'best.pt'):
            path = directory / name
            if not path.is_file():
                print(name, 'MISSING')
                continue
            state = torch.load(path, map_location='cpu', weights_only=True)
            config = state.get('config', {})
            print(json.dumps({'file': str(path), 'epoch': state.get('epoch'),
                              'best_mse': state.get('best_mse'),
                              'model': config.get('model_type'),
                              'latent': config.get('latent_channels'),
                              'K': config.get('ssai_dimension'),
                              'degradation': config.get('degradation'),
                              'upsampler': config.get('upsampler')}, ensure_ascii=False))
            del state
    elif args.command == 'status':
        state = read(BASE / 'state.json', {})
        state['controller_online'] = online()
        if state.get('log'):
            path = Path(state['log'])
            if path.exists():
                state.update(log_bytes=path.stat().st_size, log_modified=path.stat().st_mtime)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    elif args.command == 'logs':
        state = read(BASE / 'state.json', {})
        if not state.get('log'):
            raise RuntimeError('No managed run log yet.')
        path = Path(state['log'])
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 16000))
            while True:
                data = f.read()
                if data:
                    print(data.decode('utf-8', errors='replace'), end='', flush=True)
                if not args.follow:
                    break
                time.sleep(1)
    else:
        submit(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # Interrupting the client/log viewer never stops the scheduled worker.
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
