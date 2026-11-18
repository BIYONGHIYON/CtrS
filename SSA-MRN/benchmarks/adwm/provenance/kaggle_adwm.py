# Kaggle notebook for the ADWM benchmark (CtrS issue #183), percent format: every "# %%" is one notebook cell.
#
# Setup:   File > Import Notebook (this file); Accelerator: GPU T4 x2; Internet: on;
#          Add Input: dataset ctrs-pan-sharpening-dataset (train/valid QB·GF2, valid WV3, test RR + OrigScale QB·GF2·WV3·WV2).
# Run:     Save Version > Save & Run All (runs in the background, up to 12 h).
# Next:    Add Input > the previous version's Notebook Output, then Save & Run All again. Every step skips what an earlier
#          session finished and training resumes from latest.pt, so only the last version's output has to be downloaded.
#
# Steps
#   1. author : choose the LayerNorm reading of the WV3 author weights on WV3 validation (model.py --select-author),
#               then RR + FR test metrics for WV3 and the WV2 transfer, and the RR predictions zip for MATLAB.
#   2. retrain: QB (200 epochs, GPU 0) and GF2 (500 epochs, GPU 1) in parallel, seed 42, stopping before the session limit.
#   3. finished retrain runs: RR + FR metrics and RR predictions zip.
#   4. /kaggle/working/adwm_out has the layout of SSA-MRN/benchmarks/adwm/ (results/, checkpoints/, provenance/) plus
#      predictions/ (zips for the Release) and progress.json.
# Every data file is checked against the SHA-256 of the team's pinned PanCollection files (experiments 53 and 54 config.json),
# the same files the MATLAB workflow downloads; a file that does not match is not used.

# %%
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = time.time()
SESSION_HOURS = 11.0                     # training stops before this; evaluation and saving the output need the rest of 12 h
DEADLINE = T0 + SESSION_HOURS * 3600
BRANCH = 'adwm-benchmark'
REPO = 'https://github.com/BIYONGHIYON/CtrS.git'
W, TMP = Path('/kaggle/working'), Path('/tmp/ctrs')
OUT, DATA = W / 'adwm_out', TMP / 'data'
EPOCHS, SEED = {'QB': 200, 'GF2': 500}, 42
KST = timezone(timedelta(hours=9))
for p in (OUT, TMP, DATA): p.mkdir(parents=True, exist_ok=True)


def sh(cmd, cwd=None, env=None, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True, capture_output=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if check and r.returncode: raise RuntimeError(f'failed ({r.returncode}): {cmd}')
    return r.stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def now(): return datetime.now(KST).isoformat(timespec='seconds')


sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
import torch
GPUS = torch.cuda.device_count()
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', GPUS)
assert GPUS >= 1, 'turn on the GPU accelerator (T4 x2)'

# %%
# 1. outputs of the previous session (attached as input) -> /kaggle/working/adwm_out
def session_of(folder):
    p = folder / 'progress.json'
    return json.loads(p.read_text())['session'] if p.is_file() else 0

previous = sorted(set(Path('/kaggle/input').glob('*/adwm_out')) | set(Path('/kaggle/input').glob('*/*/adwm_out')), key=session_of)
if previous and not (OUT / 'progress.json').exists():
    shutil.copytree(previous[-1], OUT, dirs_exist_ok=True)
    print('continuing from', previous[-1], 'session', session_of(previous[-1]))
PROGRESS = json.loads((OUT / 'progress.json').read_text()) if (OUT / 'progress.json').exists() else dict(session=0, sessions=[])
PROGRESS['session'] += 1
SESSION = dict(session=PROGRESS['session'], started=now(), steps={})
PROGRESS['sessions'].append(SESSION)


def save_progress():
    SESSION['updated'] = now()
    (OUT / 'progress.json').write_text(json.dumps(PROGRESS, ensure_ascii=False, indent=1) + '\n')

save_progress()

# %%
# 2. code: CtrS branch (sparse) and the official ADWM code (submodule at the fixed commit)
CTRS = TMP / 'CtrS'
if not CTRS.exists():
    sh(f'git clone -q --depth 1 --branch {BRANCH} --filter=blob:none --sparse {REPO} {CTRS}')
    sh('git sparse-checkout set --no-cone /SSA-MRN/common/ /SSA-MRN/docs/images/c2ms_sp/ /SSA-MRN/benchmarks/adwm/ '
       '/SSA-MRN/experiments/performance/53_scale_mixed_training/config.json '
       '/SSA-MRN/experiments/performance/54_multi_sensor_scale_mixed_training/config.json', cwd=CTRS)
    sh('git submodule update --init --depth 1 SSA-MRN/benchmarks/adwm/upstream', cwd=CTRS, check=False)
    sh('git submodule update --init SSA-MRN/benchmarks/adwm/upstream', cwd=CTRS)   # full fetch if the shallow one missed the commit
BENCH = CTRS / 'SSA-MRN/benchmarks/adwm'
CONFIG = json.loads((BENCH / 'config.json').read_text())
CTRS_COMMIT = sh('git rev-parse HEAD', cwd=CTRS).strip()
UP_COMMIT = sh('git rev-parse HEAD', cwd=BENCH / 'upstream').strip()
assert UP_COMMIT == CONFIG['code']['commit'], f'upstream {UP_COMMIT} != config {CONFIG["code"]["commit"]}'
AUTHOR = BENCH / CONFIG['author_weights']['WV3']['path']
assert sha256(AUTHOR) == CONFIG['author_weights']['WV3']['sha256'], 'author weight file differs from config.json'
# results/ and checkpoints/ of the benchmark folder live in OUT, so everything the evaluator writes is kept in the output
for sub in ('results', 'checkpoints'):
    (OUT / sub).mkdir(exist_ok=True)
    if (BENCH / sub).is_dir() and not (BENCH / sub).is_symlink():
        shutil.copytree(BENCH / sub, OUT / sub, dirs_exist_ok=True); shutil.rmtree(BENCH / sub)
    if not (BENCH / sub).exists(): os.symlink(OUT / sub, BENCH / sub)
print('CtrS', CTRS_COMMIT, '| ADWM', UP_COMMIT)
SESSION.update(ctrs_commit=CTRS_COMMIT, adwm_commit=UP_COMMIT); save_progress()

# %%
# 3. data: Kaggle copies checked against the team's pinned PanCollection SHA-256, linked into one folder
PERF = CTRS / 'SSA-MRN/experiments/performance'
c53 = json.loads((PERF / '53_scale_mixed_training/config.json').read_text())['data_sha256']
c54 = json.loads((PERF / '54_multi_sensor_scale_mixed_training/config.json').read_text())
EXPECTED = {'train_qb.h5': c53['train'], 'valid_qb.h5': c53['val'],
            'test_qb_multiExm1.h5': c53['RR'], 'test_qb_OrigScale_multiExm1.h5': c53['FR'],
            'valid_wv3.h5': c54['data_sha256']['WV3']['val'],
            'test_wv2_multiExm1.h5': c54['transfer_data_sha256']['WV2']['RR'],
            'test_wv2_OrigScale_multiExm1.h5': c54['transfer_data_sha256']['WV2']['FR']}
for s in ('GF2', 'WV3'):
    d, low = c54['data_sha256'][s], s.lower()
    EXPECTED.update({f'test_{low}_multiExm1.h5': d['RR'], f'test_{low}_OrigScale_multiExm1.h5': d['FR']})
    if s == 'GF2': EXPECTED.update({'train_gf2.h5': d['train'], 'valid_gf2.h5': d['val']})
found = [p for p in Path('/kaggle/input').rglob('*.h5') if 'adwm_out' not in p.parts]
DATA_RECORD, AVAILABLE = {}, set()
for name, want in EXPECTED.items():
    hits = sorted(p for p in found if p.name == name or p.name.endswith('__' + name))
    match = next((p for p in hits if sha256(p) == want), None)
    DATA_RECORD[name] = dict(expected_sha256=want, candidates=[str(p) for p in hits], used=str(match) if match else None)
    if match:
        AVAILABLE.add(name)
        if not (DATA / name).exists(): os.symlink(match, DATA / name)
    print(f'{name:34s}', 'OK' if match else ('SHA-256 MISMATCH' if hits else 'MISSING'), flush=True)
(OUT / 'provenance').mkdir(exist_ok=True)
(OUT / 'provenance/data_check.json').write_text(json.dumps(DATA_RECORD, indent=1) + '\n')


def have(*names):
    missing = [n for n in names if n not in AVAILABLE]
    if missing: print('skipped, data missing or not the pinned file:', missing, flush=True)
    return not missing

# %%
# helpers: evaluator (RR + FR metrics) and RR predictions (output_mulExm_<i>.mat, key 'sr', DN, H x W x C) for MATLAB
import numpy as np, scipy.io as sio
sys.path.insert(0, str(CTRS / 'SSA-MRN'))
from common import benchmark_evaluate as BE
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
EVAL = f'{sys.executable} {CTRS}/SSA-MRN/common/benchmark_evaluate.py --model adwm --data {DATA} --device cuda'
PRED = OUT / 'predictions'; PRED.mkdir(exist_ok=True)
PRED_RECORD = json.loads((PRED / 'predictions.json').read_text()) if (PRED / 'predictions.json').exists() else {}


def export_rr(setting, sensor, seed=None, checkpoint=None):
    name = f'adwm_{setting}_{sensor}_rr.zip'
    if name in PRED_RECORD and (PRED / name).is_file(): return
    stem, scale = BE.SENSORS[sensor][0], BE.SENSORS[sensor][1]
    net, ck = BE.load_model(BENCH, setting, sensor, seed, checkpoint, 'cuda')
    folder = TMP / f'pred_{setting}_{sensor}'; shutil.rmtree(folder, ignore_errors=True); folder.mkdir()
    for i, s in BE.scenes(DATA / f'test_{stem}_multiExm1.h5', ('pan', 'lms', 'ms'), None):
        sio.savemat(folder / f'output_mulExm_{i}.mat', {'sr': BE.predict(net, s, scale, 'cuda').transpose(1, 2, 0)})
    with zipfile.ZipFile(PRED / name, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.glob('*.mat'), key=lambda p: int(p.stem.split('_')[-1])): z.write(p, p.name)
    PRED_RECORD[name] = dict(setting=setting, sensor=sensor, seed=seed, scenes=len(list(folder.glob('*.mat'))), bytes=(PRED / name).stat().st_size,
                             sha256=sha256(PRED / name), checkpoint=str(ck), checkpoint_sha256=sha256(ck), created=now())
    (PRED / 'predictions.json').write_text(json.dumps(PRED_RECORD, indent=1) + '\n')
    print('wrote', name, PRED_RECORD[name]['sha256'][:16], flush=True)


def metrics_done(setting, sensor, seed=None):
    p = OUT / 'results' / BE.run_name(setting, sensor, seed) / 'metrics.json'
    return p.is_file() and json.loads(p.read_text())['complete']

# %%
# 4. author: LayerNorm reading chosen on WV3 validation (never on test), then WV3 and WV2 transfer
try:
    if not (OUT / 'results/author_selection.json').is_file() and have('valid_wv3.h5'):
        sh(f'{sys.executable} {BENCH}/model.py --select-author --val {DATA}/valid_wv3.h5 --device cuda')
    if (OUT / 'results/author_selection.json').is_file():
        for sensor in ('WV3', 'WV2'):
            low = sensor.lower()
            if not have(f'test_{low}_multiExm1.h5', f'test_{low}_OrigScale_multiExm1.h5'): continue
            if not metrics_done('author', sensor): sh(f'{EVAL} --setting author --sensor {sensor} --checkpoint {AUTHOR}')
            export_rr('author', sensor, None, AUTHOR)
        SESSION['steps']['author'] = 'done'
    print(json.loads((OUT / 'results/author_selection.json').read_text()) if (OUT / 'results/author_selection.json').is_file() else 'no selection')
except Exception as e:
    SESSION['steps']['author'] = f'error: {e}'; print('author step failed:', e, flush=True)
save_progress()

# %%
# 5. retrain QB (GPU 0) and GF2 (GPU 1) in parallel, resumable
def run_dir(sensor): return OUT / 'checkpoints' / f'retrain_{sensor}_s{SEED}'


def epochs_done(sensor):
    h = run_dir(sensor) / 'history.json'
    return json.loads(h.read_text())['epochs'][-1]['epoch'] if h.is_file() else 0


procs = {}
for gpu, sensor in enumerate(('QB', 'GF2')):
    low = sensor.lower()
    if epochs_done(sensor) >= EPOCHS[sensor] or not have(f'train_{low}.h5', f'valid_{low}.h5'): continue
    run = run_dir(sensor); run.mkdir(parents=True, exist_ok=True)
    cmd = (f'{sys.executable} {BENCH}/train.py --sensor {sensor} --train {DATA}/train_{low}.h5 --val {DATA}/valid_{low}.h5 '
           f'--seed {SEED} --device cuda --out {run} --deadline {DEADLINE}' + (' --resume' if (run / 'latest.pt').is_file() else ''))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % GPUS), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    print('$', cmd, f'(GPU {gpu % GPUS})', flush=True)
    procs[sensor] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(TMP / f'train_{sensor}.out', 'a'), stderr=subprocess.STDOUT)
while procs:
    time.sleep(600)
    for sensor, p in list(procs.items()):
        tail = (TMP / f'train_{sensor}.out').read_text().strip().splitlines()[-1:]
        print(f'{(time.time() - T0) / 3600:.1f} h', sensor, tail, flush=True)
        if p.poll() is not None:
            print(sensor, 'exit', p.returncode, flush=True); procs.pop(sensor)
            if p.returncode: print((TMP / f'train_{sensor}.out').read_text()[-3000:], flush=True)
    SESSION['steps']['retrain'] = {s: f'{epochs_done(s)}/{EPOCHS[s]}' for s in EPOCHS}; save_progress()
SESSION['steps']['retrain'] = {s: f'{epochs_done(s)}/{EPOCHS[s]}' for s in EPOCHS}; save_progress()

# %%
# 6. finished retrain runs: history into results/, RR + FR metrics, RR predictions
for sensor in ('QB', 'GF2'):
    low = sensor.lower()
    if epochs_done(sensor) < EPOCHS[sensor]:
        print(sensor, f'{epochs_done(sensor)}/{EPOCHS[sensor]} epochs: attach this output and run again'); continue
    if not have(f'test_{low}_multiExm1.h5', f'test_{low}_OrigScale_multiExm1.h5'): continue
    try:
        res = OUT / 'results' / f'retrain_{sensor}_s{SEED}'; res.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir(sensor) / 'history.json', res / 'history.json')
        if not metrics_done('retrain', sensor, SEED): sh(f'{EVAL} --setting retrain --sensor {sensor} --seed {SEED}')
        export_rr('retrain', sensor, SEED)
        SESSION['steps'][f'retrain_{sensor}_eval'] = 'done'
    except Exception as e:
        SESSION['steps'][f'retrain_{sensor}_eval'] = f'error: {e}'; print(sensor, 'evaluation failed:', e, flush=True)
save_progress()

# %%
# 7. summary, environment, status
sh(f'{EVAL} --summary', check=False)
SESSION.update(ended=now(), hours=round((time.time() - T0) / 3600, 2), torch=torch.__version__, cuda=torch.version.cuda,
               gpus=[torch.cuda.get_device_name(i) for i in range(GPUS)], python=sys.version.split()[0])
save_progress()
print(json.dumps(SESSION, ensure_ascii=False, indent=1))
need = [f'adwm_author_{s}_rr.zip' for s in ('WV3', 'WV2')] + [f'adwm_retrain_{s}_rr.zip' for s in ('QB', 'GF2')]
left = [n for n in need if n not in PRED_RECORD]
print('ALL DONE: download /kaggle/working/adwm_out' if not left else f'not finished yet: {left} -> Add Input (this output) and Save & Run All')
sh(f'du -sh {OUT}; find {OUT} -maxdepth 3 | sort | head -80')
