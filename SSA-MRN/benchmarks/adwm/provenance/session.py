"""One Kaggle session of the ADWM benchmark (CtrS issue #183): resume the most advanced saved state, do what is left, save.

Training (QB 200, GF2 500 epochs) does not fit in one Kaggle session, so it runs over several sessions of --hours each.
Every session runs the same notebook; each step skips what an earlier session finished:
    1. author : choose the LayerNorm reading of the WV3 author weights on WV3 validation (model.py --select-author),
                RR + FR test metrics for WV3 and the WV2 transfer, RR predictions zip for MATLAB
    2. retrain runs that finished at the end of the previous session: evaluate (same as 4)
    3. retrain QB and GF2 (train.py --resume --deadline, --checkpointing auto). Two GPUs: one run per GPU in parallel.
       One GPU: the run with more time left first, then the other if time remains. Training stops --stop-minutes before
       the end of the session (train.py does not start an epoch that would end after that).
    4. retrain runs that finished now: history into results/, RR + FR metrics, RR predictions zip
    5. results/summary.json (common/benchmark_evaluate.py --summary)
    6. progress.json: this session (platform, GPUs, LAConv recomputation, code, data, epochs, steps) and what to do next

State (--state, the notebook output /kaggle/working/adwm_out) has the layout of SSA-MRN/benchmarks/adwm/ plus:
    results/        metrics, author_selection.json, summary.json, retrain histories    (benchmark results/ links here)
    checkpoints/    retrain_<QB|GF2>_s42/{best,latest,initial}.pt, history.json        (benchmark checkpoints/ links here)
    predictions/    adwm_<setting>_<sensor>_rr.zip (output_mulExm_<i>.mat, key sr, DN, H x W x C) and predictions.json
    provenance/     data_check.json (pinned SHA-256 of every data file), code.json
    logs/           train_<sensor>_session<n>.log
    progress.json   one entry per session
Choosing the state to resume: every folder under /kaggle/input with a progress.json (the previous version's output attached
as input, or an adwm_out zip uploaded as a dataset, which Kaggle extracts) and every adwm_out*.zip there is a candidate.
The most advanced one wins: QB + GF2 epochs, then finished prediction zips, then session number. If one candidate is ahead
in QB and another in GF2, the same state was trained in two places at once; the session warns and still takes the larger
total. Data: every h5 is checked against the SHA-256 of the team's pinned files (performance experiments 53 and 54
config.json); a file that does not match is not used. Code: model.py and train.py must not change while training is
unfinished (their SHA-256 is compared with the previous session; training is skipped if they differ).

Kaggle notebook (Accelerator GPU T4 x2, Internet on, Input: dataset "CtrS PAN Sharpening Dataset"); one code cell:

    !rm -rf /tmp/ctrs && mkdir -p /tmp/ctrs
    !git clone -q --depth 1 --branch adwm-benchmark --filter=blob:none --sparse https://github.com/BIYONGHIYON/CtrS.git /tmp/ctrs/CtrS
    !cd /tmp/ctrs/CtrS && git sparse-checkout set --no-cone /.gitmodules /SSA-MRN/common/ /SSA-MRN/benchmarks/adwm/ \\
        /SSA-MRN/docs/images/c2ms_sp/baseline_metrics.json \\
        /SSA-MRN/experiments/performance/53_scale_mixed_training/config.json \\
        /SSA-MRN/experiments/performance/54_multi_sensor_scale_mixed_training/config.json
    !cd /tmp/ctrs/CtrS && git submodule update --init SSA-MRN/benchmarks/adwm/upstream
    !pip install -q einops   # imported by the official model file (usually already in the Kaggle image)
    !python /tmp/ctrs/CtrS/SSA-MRN/benchmarks/adwm/provenance/session.py --hours 6

First session: Save Version > Save & Run All (runs in the background). Next sessions: Add Input > the previous version's
output (this notebook) and Save & Run All again, until the log ends with ALL DONE; then download adwm_out from the output.

Local smoke test (synthetic data, CPU):
    python session.py --inputs <folder> --state <state> --data <links> --expected <sha.json> --device cpu \\
        --epochs QB=3,GF2=4 --train-extra "--limit 64" --poll 2
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
SSA = BENCH.parents[1]
CTRS = SSA.parent
KST = timezone(timedelta(hours=9))
SEED = 42
EPOCHS = {'QB': 200, 'GF2': 500}
SENSORS = ('QB', 'GF2')
NEEDED_ZIPS = ['adwm_author_WV3_rr.zip', 'adwm_author_WV2_rr.zip', 'adwm_retrain_QB_rr.zip', 'adwm_retrain_GF2_rr.zip']
PINNED_CODE = ('model.py', 'train.py')


def now(): return datetime.now(KST).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def sh(cmd, check=True, cwd=None):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if r.stdout.strip(): print(r.stdout[-6000:], flush=True)
    if r.stderr.strip(): print(r.stderr[-6000:], flush=True)
    if check and r.returncode: raise RuntimeError(f'exit {r.returncode}: {cmd}')
    return r.stdout


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError): return default


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    tmp.replace(path)


# ---------------------------------------------------------------- state candidates
class Candidate:
    """A saved state: a folder with progress.json, or a zip containing one (possibly under a top folder)."""

    def __init__(self, source):
        self.source = Path(source); self.is_zip = self.source.suffix == '.zip'
        if self.is_zip:
            self.zip = zipfile.ZipFile(self.source)
            names = [n for n in self.zip.namelist() if n.endswith('progress.json')]
            if not names: raise ValueError('no progress.json')
            self.prefix = min(names, key=len)[:-len('progress.json')]
            self.names = set(self.zip.namelist())
        progress = self.json('progress.json') or {}
        self.session = progress.get('session', 0)
        self.epochs = {}
        for s in SENSORS:
            h = self.json(f'checkpoints/retrain_{s}_s{SEED}/history.json')
            self.epochs[s] = h['epochs'][-1]['epoch'] if h and h.get('epochs') else 0
        preds = self.json('predictions/predictions.json') or {}
        self.zips = sorted(n for n in preds if self.exists(f'predictions/{n}'))
        self.key = (sum(self.epochs.values()), len(self.zips), self.session)

    def json(self, rel):
        if not self.is_zip: return read_json(self.source / rel)
        name = self.prefix + rel
        return json.loads(self.zip.read(name)) if name in self.names else None

    def exists(self, rel): return (self.prefix + rel) in self.names if self.is_zip else (self.source / rel).is_file()

    def restore(self, dest):
        if self.is_zip:
            for info in self.zip.infolist():
                if not info.filename.startswith(self.prefix) or info.is_dir(): continue
                rel = info.filename[len(self.prefix):]
                if rel.startswith('/') or '..' in Path(rel).parts: raise ValueError(f'unsafe path in zip: {info.filename}')
                target = dest / rel; target.parent.mkdir(parents=True, exist_ok=True)
                with self.zip.open(info) as src, open(target, 'wb') as out: shutil.copyfileobj(src, out)
        else:
            shutil.copytree(self.source, dest, symlinks=False, dirs_exist_ok=True)

    def __str__(self):
        return (f"{self.source} (session {self.session}, QB {self.epochs['QB']}, GF2 {self.epochs['GF2']} epochs, "
                f"{len(self.zips)} prediction zips)")


def find_candidates(inputs, state, depth=5):
    found = []
    if (state / 'progress.json').is_file(): found.append(state)
    if inputs.is_dir():
        base = len(inputs.parts)
        for root, dirs, files in os.walk(inputs):
            root = Path(root)
            if len(root.parts) - base >= depth: dirs[:] = []
            if 'progress.json' in files: found.append(root); dirs[:] = []; continue
            found += [root / f for f in files if f.startswith('adwm_out') and f.endswith('.zip')]
    out = []
    for f in found:
        try: out.append(Candidate(f))
        except Exception as e: print(f'not a state, ignored: {f} ({e})', flush=True)
    return out


def choose_state(cands, state):
    warnings = []
    for a in cands:
        for b in cands:
            if a.epochs['QB'] > b.epochs['QB'] and a.epochs['GF2'] < b.epochs['GF2']:
                warnings.append(f'QB is further in {a.source} but GF2 in {b.source}: the state seems to have been trained in two '
                                'places at once. Only one place may run at a time; the larger total is used.')
    if not cands: return None, warnings
    # ties: the current state first, then folders before zips
    best = max(cands, key=lambda c: (c.key, c.source == state, not c.is_zip))
    return best, sorted(set(warnings))


# ---------------------------------------------------------------- session
class Session:
    def __init__(self, a):
        self.a, self.t0 = a, time.time()
        self.end = self.t0 + a.hours * 3600
        self.train_deadline = self.end - a.stop_minutes * 60
        self.state, self.data = a.state.resolve(), a.data
        self.device = a.device
        self.epochs = dict(EPOCHS, **a.epochs)
        self.smoke = bool(a.epochs or a.train_extra or a.expected)

    # ---- bookkeeping
    def save(self):
        self.entry['updated'] = now(); self.entry['hours'] = round((time.time() - self.t0) / 3600, 2)
        write_json(self.state / 'progress.json', self.progress)

    @contextmanager
    def step(self, name):
        print(f'\n===== {name} =====', flush=True)
        try:
            yield
            self.entry['steps'].setdefault(name, 'done')
        except Exception as e:
            traceback.print_exc()
            self.entry['steps'][name] = f'error: {type(e).__name__}: {e}'
        self.save()

    # ---- 0. state, code, data
    def restore_state(self):
        cands = find_candidates(self.a.inputs, self.state)
        for c in cands: print('candidate:', c, flush=True)
        best, warnings = choose_state(cands, self.state)
        for w in warnings: print('WARNING:', w, flush=True)
        if best is None: print('no saved state: starting from the beginning', flush=True)
        elif best.source != self.state:
            if self.state.exists() and any(self.state.iterdir()):
                backup = Path(self.a.backup_dir) / f'adwm_out_before_{datetime.now(KST):%Y%m%d_%H%M%S}'
                shutil.move(str(self.state), str(backup)); print('previous working state moved to', backup, flush=True)
            self.state.mkdir(parents=True, exist_ok=True); best.restore(self.state)
            print('resuming', best, flush=True)
        else: print('resuming the state already in', self.state, flush=True)
        self.state.mkdir(parents=True, exist_ok=True)
        self.progress = read_json(self.state / 'progress.json', dict(model='adwm', session=0, sessions=[]))
        self.previous = self.progress['sessions'][-1] if self.progress['sessions'] else None
        self.progress['session'] += 1
        self.entry = dict(session=self.progress['session'], started=now(), platform=self.a.platform, smoke_test=self.smoke,
                          state_source=str(best.source) if best else None, state_warnings=warnings, steps={})
        self.progress['sessions'].append(self.entry); self.save()

    def link_benchmark_folders(self):
        """results/ and checkpoints/ of the benchmark folder live in the state, so everything written there is kept."""
        for sub in ('results', 'checkpoints'):
            target, link = self.state / sub, BENCH / sub
            target.mkdir(parents=True, exist_ok=True)
            if link.is_symlink(): link.unlink()
            elif link.is_dir():                       # files committed to the branch: copied only where the state has none
                for f in link.rglob('*'):
                    if f.is_file() and not (target / f.relative_to(link)).exists():
                        (target / f.relative_to(link)).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, target / f.relative_to(link))
                shutil.rmtree(link)
            os.symlink(target, link)

    def check_code(self):
        import torch
        config = read_json(BENCH / 'config.json')
        up = sh(f'git -C {BENCH / "upstream"} rev-parse HEAD').strip()
        if up != config['code']['commit']: raise SystemExit(f'upstream {up} != config.json {config["code"]["commit"]}')
        self.author_weights = BENCH / config['author_weights']['WV3']['path']
        if sha256(self.author_weights) != config['author_weights']['WV3']['sha256']: raise SystemExit('author weight file differs from config.json')
        ctrs = sh(f'git -C {CTRS} rev-parse HEAD', check=False).strip() or None
        code = {f: sha256(BENCH / f) for f in PINNED_CODE + ('provenance/session.py',)}
        self.code_changed = []
        if self.previous and self.previous.get('code'):
            self.code_changed = [f for f in PINNED_CODE if self.previous['code'].get(f) != code[f]]
            if self.previous['code'].get('provenance/session.py') != code['provenance/session.py']:
                print('note: session.py changed since the previous session (does not affect training)', flush=True)
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        self.entry.update(ctrs_commit=ctrs, adwm_commit=up, code=code, code_changed=self.code_changed, python=sys.version.split()[0],
                          torch=torch.__version__, cuda=torch.version.cuda, gpus=gpus, device=self.device)
        print(f'CtrS {ctrs} | ADWM {up} | torch {torch.__version__} | GPUs {gpus or "none"}', flush=True)
        if self.code_changed: print('WARNING: changed since the previous session:', self.code_changed, flush=True)

    def expected_data(self):
        if self.a.expected: return read_json(self.a.expected)
        perf = SSA / 'experiments/performance'
        c53 = read_json(perf / '53_scale_mixed_training/config.json')['data_sha256']
        c54 = read_json(perf / '54_multi_sensor_scale_mixed_training/config.json')
        exp = {'train_qb.h5': c53['train'], 'valid_qb.h5': c53['val'],
               'test_qb_multiExm1.h5': c53['RR'], 'test_qb_OrigScale_multiExm1.h5': c53['FR'],
               'valid_wv3.h5': c54['data_sha256']['WV3']['val'],
               'test_wv2_multiExm1.h5': c54['transfer_data_sha256']['WV2']['RR'],
               'test_wv2_OrigScale_multiExm1.h5': c54['transfer_data_sha256']['WV2']['FR']}
        for s in ('GF2', 'WV3'):
            d, low = c54['data_sha256'][s], s.lower()
            exp.update({f'test_{low}_multiExm1.h5': d['RR'], f'test_{low}_OrigScale_multiExm1.h5': d['FR']})
        exp.update({'train_gf2.h5': c54['data_sha256']['GF2']['train'], 'valid_gf2.h5': c54['data_sha256']['GF2']['val']})
        return exp

    def check_data(self):
        expected = self.expected_data(); self.data.mkdir(parents=True, exist_ok=True)
        found = []
        for root, dirs, files in os.walk(self.a.inputs, followlinks=True):
            dirs[:] = [d for d in dirs if not d.startswith('adwm_out')]
            found += [Path(root) / f for f in files if f.endswith('.h5')]
        record, self.available = {}, set()
        for name, want in expected.items():
            hits = sorted(p for p in found if p.name == name or p.name.endswith('__' + name))
            match = next((p for p in hits if sha256(p) == want), None)
            record[name] = dict(expected_sha256=want, candidates=[str(p) for p in hits], used=str(match) if match else None)
            link = self.data / name
            if link.is_symlink() or link.exists(): link.unlink()
            if match: self.available.add(name); os.symlink(match, link)
            print(f'{name:34s}', 'OK' if match else ('SHA-256 MISMATCH, not used' if hits else 'MISSING'), flush=True)
        write_json(self.state / 'provenance/data_check.json', dict(checked=now(), files=record))
        self.entry['data'] = f'{len(self.available)}/{len(expected)} files match the pinned SHA-256'
        if not self.available:
            self.save()
            raise SystemExit(f'no data file under {self.a.inputs} matches the pinned SHA-256: attach the dataset '
                             '"CtrS PAN Sharpening Dataset" (Add Input) and run again')

    def have(self, *names):
        missing = [n for n in names if n not in self.available]
        if missing: print('skipped, data missing or not the pinned file:', missing, flush=True)
        return not missing

    # ---- evaluation helpers
    def evaluator(self):
        if not hasattr(self, '_be'):
            sys.path.insert(0, str(SSA))
            from common import benchmark_evaluate
            self._be = benchmark_evaluate
        return self._be

    def eval_cmd(self, setting, sensor, seed=None, checkpoint=None):
        cmd = (f'{shlex.quote(sys.executable)} {SSA}/common/benchmark_evaluate.py --model adwm --setting {setting} --sensor {sensor} '
               f'--data {self.data} --device {self.device}')
        if seed is not None: cmd += f' --seed {seed}'
        if checkpoint: cmd += f' --checkpoint {checkpoint}'
        return cmd

    def metrics_done(self, setting, sensor, seed=None):
        m = read_json(self.state / 'results' / self.evaluator().run_name(setting, sensor, seed) / 'metrics.json')
        return bool(m and m.get('complete'))

    def predictions(self): return read_json(self.state / 'predictions/predictions.json', {})

    def zip_done(self, name):
        rec, path = self.predictions().get(name), self.state / 'predictions' / name
        return bool(rec and path.is_file() and sha256(path) == rec['sha256'])

    def export_rr(self, setting, sensor, seed=None, checkpoint=None):
        """RR test predictions for MATLAB: output_mulExm_<i>.mat with key sr, DN, H x W x C, in one zip."""
        import numpy as np
        import scipy.io as sio
        import torch
        name = f'adwm_{setting}_{sensor}_rr.zip'
        if self.zip_done(name): print(name, 'already done', flush=True); return
        BE = self.evaluator(); stem, scale = BE.SENSORS[sensor][0], BE.SENSORS[sensor][1]
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        net, ck = BE.load_model(BENCH, setting, sensor, seed, checkpoint, self.device)
        rr = self.data / f'test_{stem}_multiExm1.h5'
        folder = Path(self.a.backup_dir) / f'pred_{setting}_{sensor}'; shutil.rmtree(folder, ignore_errors=True); folder.mkdir(parents=True)
        n = 0
        for i, s in BE.scenes(rr, ('pan', 'lms', 'ms'), None):
            out = BE.predict(net, s, scale, self.device).transpose(1, 2, 0)
            assert out.shape == s['lms'].transpose(1, 2, 0).shape and np.isfinite(out).all()
            sio.savemat(folder / f'output_mulExm_{i}.mat', {'sr': out}); n += 1
        target = self.state / 'predictions' / name; target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for i in range(n): z.write(folder / f'output_mulExm_{i}.mat', f'output_mulExm_{i}.mat')
        tmp.replace(target); shutil.rmtree(folder, ignore_errors=True)
        rec = self.predictions()
        rec[name] = dict(setting=setting, sensor=sensor, seed=seed, scenes=n, format='output_mulExm_<i>.mat, key sr, DN, H x W x C',
                         bytes=target.stat().st_size, sha256=sha256(target),
                         checkpoint=Path(ck).relative_to(CTRS).as_posix() if Path(ck).is_relative_to(CTRS) else str(ck),
                         checkpoint_sha256=sha256(ck), data=dict(file=rr.name, sha256=sha256(rr)), session=self.entry['session'], created=now())
        write_json(self.state / 'predictions/predictions.json', rec)
        print('wrote', name, rec[name]['sha256'][:16], f'{n} scenes', flush=True)

    # ---- 1. author
    def author(self):
        sel = self.state / 'results/author_selection.json'
        if not sel.is_file():
            if not self.have('valid_wv3.h5'): self.entry['steps']['1_author'] = 'skipped: valid_wv3.h5'; return
            sh(f'{shlex.quote(sys.executable)} {BENCH}/model.py --select-author --val {self.data}/valid_wv3.h5 --device {self.device}')
        print('LayerNorm reading:', read_json(sel)['chosen'], flush=True)
        for sensor in ('WV3', 'WV2'):
            low = sensor.lower()
            if not self.have(f'test_{low}_multiExm1.h5', f'test_{low}_OrigScale_multiExm1.h5'):
                self.entry['steps']['1_author'] = f'incomplete: {sensor} data'; continue
            if not self.metrics_done('author', sensor): sh(self.eval_cmd('author', sensor, checkpoint=self.author_weights))
            self.export_rr('author', sensor, None, self.author_weights)

    # ---- 2 / 4. finished retrain runs
    def run_dir(self, sensor): return self.state / 'checkpoints' / f'retrain_{sensor}_s{SEED}'

    def history(self, sensor): return read_json(self.run_dir(sensor) / 'history.json')

    def epochs_done(self, sensor):
        h = self.history(sensor)
        return h['epochs'][-1]['epoch'] if h and h.get('epochs') else 0

    def epoch_seconds(self, sensor):
        h = self.history(sensor)
        return max(r['seconds'] for r in h['epochs'][-3:]) if h and h.get('epochs') else None

    def evaluate_finished(self, step):
        for sensor in SENSORS:
            low = sensor.lower()
            if self.epochs_done(sensor) < self.epochs[sensor]: continue
            if self.metrics_done('retrain', sensor, SEED) and self.zip_done(f'adwm_retrain_{sensor}_rr.zip'): continue
            if not self.have(f'test_{low}_multiExm1.h5', f'test_{low}_OrigScale_multiExm1.h5'):
                self.entry['steps'][step] = f'incomplete: {sensor} data'; continue
            res = self.state / 'results' / f'retrain_{sensor}_s{SEED}'; res.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.run_dir(sensor) / 'history.json', res / 'history.json')
            if not self.metrics_done('retrain', sensor, SEED): sh(self.eval_cmd('retrain', sensor, SEED))
            self.export_rr('retrain', sensor, SEED)

    # ---- 3. training
    def train_cmd(self, sensor, deadline):
        low, run = sensor.lower(), self.run_dir(sensor)
        cmd = (f'{shlex.quote(sys.executable)} {BENCH}/train.py --sensor {sensor} --train {self.data}/train_{low}.h5 '
               f'--val {self.data}/valid_{low}.h5 --seed {SEED} --device {"cuda" if self.device.startswith("cuda") else self.device} '
               f'--out {run} --deadline {deadline} --session {self.entry["session"]} --checkpointing {self.a.checkpointing}')
        if self.epochs[sensor] != EPOCHS[sensor]: cmd += f' --epochs {self.epochs[sensor]}'
        if self.a.train_extra: cmd += ' ' + self.a.train_extra
        if (run / 'latest.pt').is_file(): cmd += ' --resume'
        return cmd

    def launch(self, sensor, gpu):
        run = self.run_dir(sensor); run.mkdir(parents=True, exist_ok=True)
        log = self.state / 'logs' / f'train_{sensor}_session{self.entry["session"]}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        cmd = self.train_cmd(sensor, self.train_deadline)
        print('$', cmd, f'(GPU {gpu})' if gpu is not None else '', '>', log, flush=True)
        return subprocess.Popen(cmd, shell=True, env=env, stdout=open(log, 'a'), stderr=subprocess.STDOUT), log

    def wait(self, procs):
        while procs:
            time.sleep(self.a.poll)
            for sensor, (p, log) in list(procs.items()):
                lines = log.read_text(errors='replace').strip().splitlines()
                print(f'{(time.time() - self.t0) / 3600:5.2f} h  {sensor}: {lines[-1] if lines else "(starting)"}', flush=True)
                if p.poll() is not None:
                    print(sensor, 'exit', p.returncode, flush=True); procs.pop(sensor)
                    if p.returncode: print('\n'.join(lines[-40:]), flush=True); self.entry['steps'][f'3_train_{sensor}'] = f'error: exit {p.returncode}'
            self.record_training(); self.save()

    def record_training(self):
        tr = self.entry.setdefault('training', {})
        for s in SENSORS:
            h = self.history(s); log = (h or {}).get('meta', {}).get('session_log', [])
            mine = [e for e in log if e.get('session') == self.entry['session']]
            tr[s] = dict(epochs_before=tr.get(s, {}).get('epochs_before', self.epochs_done(s)), epochs_after=self.epochs_done(s),
                         total=self.epochs[s], checkpointing=mine[-1]['checkpointing'] if mine else None,
                         gpu=mine[-1]['gpu'] if mine else None, seconds_per_epoch=self.epoch_seconds(s))

    def train(self):
        self.record_training()
        unfinished = [s for s in SENSORS if self.epochs_done(s) < self.epochs[s]]
        todo = [s for s in unfinished if self.have(f'train_{s.lower()}.h5', f'valid_{s.lower()}.h5')]
        if len(todo) < len(unfinished): self.entry['steps']['3_train'] = f'incomplete: no data for {sorted(set(unfinished) - set(todo))}'
        if not todo: print('nothing to train', flush=True); return
        started = [s for s in todo if self.epochs_done(s) > 0]
        if self.code_changed and started and not self.a.allow_code_change:
            raise RuntimeError(f'{self.code_changed} changed while training is unfinished ({started}); training skipped. '
                               'Restore the code of the previous session (progress.json code) or pass --allow-code-change.')
        import torch
        ngpu = torch.cuda.device_count() if self.device.startswith('cuda') else 0
        if ngpu >= 2 and len(todo) > 1:
            self.wait({s: self.launch(s, i) for i, s in enumerate(todo)})
        else:
            ests = {s: self.epoch_seconds(s) for s in todo}
            def left(s):                                             # seconds if every run has a measured epoch, else epochs
                return (self.epochs[s] - self.epochs_done(s)) * (ests[s] if all(ests.values()) else 1)
            for s in sorted(todo, key=left, reverse=True):           # more time left first
                # no epoch of this run measured yet: the other run's epoch time stands in for it
                est = self.epoch_seconds(s) or max([v for v in ests.values() if v] + [self.epoch_seconds(o) or 0 for o in SENSORS])
                if time.time() >= self.train_deadline or (est and time.time() + 1.3 * est + 60 > self.train_deadline):
                    print(f'{s}: not enough time left for one epoch' + (f' (~{est:.0f}s)' if est else '') + ', not started', flush=True)
                    continue
                self.wait({s: self.launch(s, 0 if ngpu else None)})
        self.record_training()

    # ---- 5 / 6
    def summary(self):
        sh(f'{shlex.quote(sys.executable)} {SSA}/common/benchmark_evaluate.py --model adwm --summary')

    def finish(self):
        self.record_training()
        done_zips = [n for n in NEEDED_ZIPS if self.zip_done(n)]
        left = [n for n in NEEDED_ZIPS if n not in done_zips]
        self.entry.update(ended=now(), prediction_zips=done_zips)
        self.save()
        print('\n===== session', self.entry['session'], 'summary =====', flush=True)
        print(json.dumps({k: v for k, v in self.entry.items() if k not in ('code',)}, ensure_ascii=False, indent=1), flush=True)
        if not left:
            print('\nALL DONE: download adwm_out from the notebook output (Output > adwm_out).', flush=True); return
        print('\nnot finished yet:', left, flush=True)
        session_train = max(self.a.hours * 3600 - self.a.stop_minutes * 60 - 600, 1)
        for s in SENSORS:
            est, rest = self.epoch_seconds(s), self.epochs[s] - self.epochs_done(s)
            if rest > 0 and est: print(f'  {s}: {rest} epochs left, about {rest * est / 3600:.1f} h of training '
                                       f'(~{-(-rest * est // session_train):.0f} more sessions of {self.a.hours:g} h)', flush=True)
        print('NEXT: Add Input > Notebook Output of this version (it contains adwm_out), remove older adwm_out inputs, '
              'then Save Version > Save & Run All. Run in one place at a time.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hours', type=float, default=6.0, help='session length; training stops --stop-minutes before its end')
    p.add_argument('--stop-minutes', type=float, default=35.0, help='time kept for evaluation and saving after training')
    p.add_argument('--state', type=Path, default=Path('/kaggle/working/adwm_out'), help='working state (the notebook output)')
    p.add_argument('--data', type=Path, default=Path('/tmp/ctrs/data'), help='folder for links to the checked h5 files')
    p.add_argument('--inputs', type=Path, default=Path('/kaggle/input'), help='where the dataset and earlier states are attached')
    p.add_argument('--backup-dir', default='/tmp/ctrs', help='scratch space (outside the output)')
    p.add_argument('--platform', default='kaggle')
    p.add_argument('--device', default=None, help='default cuda if available, else cpu')
    p.add_argument('--checkpointing', choices=('auto', 'on', 'off'), default='auto', help='LAConv recomputation (train.py)')
    p.add_argument('--poll', type=float, default=600, help='seconds between training progress lines')
    p.add_argument('--allow-code-change', action='store_true', help='train even if model.py / train.py changed mid-training')
    p.add_argument('--expected', type=Path, help='smoke test only: JSON {file name: sha256} instead of the pinned data')
    p.add_argument('--epochs', default='', help='smoke test only: QB=3,GF2=4')
    p.add_argument('--train-extra', default='', help='smoke test only: extra train.py arguments, e.g. "--limit 64"')
    a = p.parse_args()
    a.epochs = {k: int(v) for k, v in (x.split('=') for x in a.epochs.split(',') if x)}
    if a.device is None:
        import torch
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(a.backup_dir).mkdir(parents=True, exist_ok=True)

    s = Session(a)
    s.restore_state()
    s.link_benchmark_folders()
    s.check_code()
    s.check_data()
    s.save()
    with s.step('1_author'): s.author()
    with s.step('2_evaluate_previous'): s.evaluate_finished('2_evaluate_previous')
    with s.step('3_train'): s.train()
    with s.step('4_evaluate'): s.evaluate_finished('4_evaluate')
    with s.step('5_summary'): s.summary()
    s.finish()


if __name__ == '__main__':
    main()
