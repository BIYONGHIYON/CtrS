# Kaggle notebook for the ADWM benchmark (CtrS), percent format: each "# %%" is one notebook cell.
# Import with File > Import Notebook. Inputs: dataset ctrs-pan-sharpening-dataset (PanCollection train/valid/test H5);
# accelerator GPU T4 x2; internet on.
#
# 1. author : WV3 author weights (no training) -> RR+FR metrics for WV3 and WV2 transfer, RR predictions for MATLAB
# 2. retrain: QB (200 epochs) on GPU 0 and GF2 (500 epochs) on GPU 1 in parallel, seed 42, resumable; stops before the 12 h limit.
#             To continue in the next session, add this notebook's previous output as an input: latest.pt is picked up automatically.
# 3. retrain runs that reached their last epoch -> RR+FR metrics and RR predictions
# Everything to bring back is collected in /kaggle/working/adwm_out (download it from the notebook output).

# %%
import os, sys, json, time, hashlib, shutil, subprocess, zipfile
from pathlib import Path
T0 = time.time()
TRAIN_HOURS = 10.8                       # training stops before this; the rest is left for evaluation and saving the output
DEADLINE = T0 + TRAIN_HOURS * 3600
BRANCH = 'adwm-benchmark'                # use 'main' after the registration PR is merged
W = Path('/kaggle/working'); TMP = Path('/tmp/ctrs'); TMP.mkdir(exist_ok=True)
OUT = W / 'adwm_out'; OUT.mkdir(exist_ok=True)


def sh(cmd, cwd=None, env=None, check=True):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True, env=env)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode: raise RuntimeError(f'failed ({r.returncode}): {cmd}')
    return r.stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
sh('pip install -q einops')
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())

# %%
# CtrS code (sparse) + official ADWM code (submodule, fixed commit)
CTRS = TMP / 'CtrS'
if not CTRS.exists():
    sh(f'git clone -q --depth 1 --branch {BRANCH} --filter=blob:none --sparse https://github.com/BIYONGHIYON/CtrS.git {CTRS}')
    sh('git sparse-checkout set SSA-MRN/common SSA-MRN/docs/images/c2ms_sp SSA-MRN/benchmarks/adwm', cwd=CTRS)
    sh('git submodule update --init SSA-MRN/benchmarks/adwm/upstream', cwd=CTRS)
CTRS_COMMIT = sh('git rev-parse HEAD', cwd=CTRS).strip()
BENCH = CTRS / 'SSA-MRN/benchmarks/adwm'
UP_COMMIT = sh('git rev-parse HEAD', cwd=BENCH / 'upstream').strip()
assert UP_COMMIT == json.loads((BENCH / 'config.json').read_text())['code']['commit'], UP_COMMIT
AUTHOR = BENCH / 'upstream/weights/BWNET_LAGConv_WV3/model.pth'
assert sha256(AUTHOR) == json.loads((BENCH / 'config.json').read_text())['author_weights']['WV3']['sha256']
print('CtrS', CTRS_COMMIT, 'ADWM', UP_COMMIT)

# %%
# data: Kaggle dataset copies linked into one folder; test files checked against public PanCollection (HF, LFS oid = SHA-256)
import urllib.request
DATA = TMP / 'data'; DATA.mkdir(exist_ok=True)
files = [p for p in Path('/kaggle/input').rglob('*.h5') if p.stat().st_size > 1 << 20]
names = [f'test_{s}{o}_multiExm1.h5' for s in ('wv3', 'wv2', 'qb', 'gf2') for o in ('', '_OrigScale')]
names += [f'{k}_{s}.h5' for s in ('qb', 'gf2') for k in ('train', 'valid')]
try:
    req = urllib.request.Request('https://huggingface.co/api/datasets/elsting/PanCollection/tree/main/test_data', headers={'User-Agent': 'generic-client'})
    HF = {Path(f['path']).name: f.get('lfs', {}).get('oid') for f in json.load(urllib.request.urlopen(req, timeout=60))}
except Exception as e:
    HF = {}; print('HF listing failed', e)
DATA_HASH = {}
for n in names:
    hits = sorted(p for p in files if (p.name == n or p.name.endswith('__' + n)) and not p.name.endswith('._' + n))
    if not hits: print('missing', n); continue
    if not (DATA / n).exists(): os.symlink(hits[0], DATA / n)
    DATA_HASH[n] = dict(source=hits[0].name, bytes=hits[0].stat().st_size, sha256=sha256(hits[0]), hf_main=HF.get(n))
    print(n, DATA_HASH[n]['sha256'][:16], 'HF', (HF.get(n) or '-')[:16], 'same' if HF.get(n) == DATA_HASH[n]['sha256'] else '')
(OUT / 'data_hashes.json').write_text(json.dumps(DATA_HASH, indent=1) + '\n')

# %%
# helpers: evaluation through the CtrS evaluator, RR predictions as per-scene .mat (DN, H x W x C, key 'sr') zipped for MATLAB
import h5py, numpy as np, scipy.io as sio
EVAL = f'{sys.executable} {CTRS}/SSA-MRN/common/benchmark_evaluate.py --folder {BENCH} --data {DATA} --device cuda'
STEM = {'QB': 'qb', 'GF2': 'gf2', 'WV3': 'wv3', 'WV2': 'wv2'}
SCALE = {'QB': 2047., 'GF2': 1023., 'WV3': 2047., 'WV2': 2047.}
BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8, 'WV2': 8}
sys.path.insert(0, str(CTRS / 'SSA-MRN'))
from common.benchmark_evaluate import load_model


def export_rr(setting, sensor, seed, checkpoint=None):
    """Write adwm_<setting>_<SENSOR>_rr.zip (20 x output_mulExm_<i>.mat, key 'sr', DN, H x W x C) for the MATLAB workflow."""
    net, ck = load_model(BENCH, setting, sensor, seed, checkpoint, 'cuda')
    folder = TMP / f'pred_{setting}_{sensor}'; shutil.rmtree(folder, ignore_errors=True); folder.mkdir()
    with h5py.File(DATA / f'test_{STEM[sensor]}_multiExm1.h5', 'r') as h:
        for i in range(h['gt'].shape[0]):
            x = [torch.from_numpy(np.asarray(h[k][i], dtype=np.float32) / SCALE[sensor])[None].cuda() for k in ('pan', 'lms', 'ms')]
            with torch.no_grad(): y = net(*x)[0].double().cpu().numpy() * SCALE[sensor]
            sio.savemat(folder / f'output_mulExm_{i}.mat', {'sr': y.transpose(1, 2, 0)})
    z = OUT / f'adwm_{setting}_{sensor}_rr.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(folder.glob('*.mat')): f.write(p, p.name)
    print('wrote', z.name, z.stat().st_size, sha256(z)[:16], flush=True)
    return dict(zip=z.name, bytes=z.stat().st_size, sha256=sha256(z), checkpoint=str(ck))


PRED = json.loads((OUT / 'predictions.json').read_text()) if (OUT / 'predictions.json').exists() else {}

# %%
# 1. author: WV3 author weights, WV3 and WV2 transfer (RR + FR), then RR predictions
for sensor in ('WV3', 'WV2'):
    if (BENCH / f'results/author_{sensor}/metrics.json').exists(): continue
    sh(f'{EVAL} --setting author --sensor {sensor} --checkpoint {AUTHOR}')
    PRED[f'author_{sensor}'] = export_rr('author', sensor, None, AUTHOR)
(OUT / 'predictions.json').write_text(json.dumps(PRED, indent=1) + '\n')
for p in sorted((BENCH / 'results').glob('author_*/metrics.json')):
    m = json.loads(p.read_text()); print(p.parent.name, {k: round(v, 4) for k, v in m['rr']['mean'].items()})
print('paper WV3: PSNR 39.170 SAM 2.913 ERGAS 2.145 Q8 0.921')

# %%
# 2. retrain QB (GPU 0) and GF2 (GPU 1) in parallel, resumable across sessions
RUNS = W / 'runs'; RUNS.mkdir(exist_ok=True)
EPOCHS = {'QB': 200, 'GF2': 500}
for prev in Path('/kaggle/input').rglob('retrain_*_s42/latest.pt'):       # previous session's output attached as input
    dst = RUNS / prev.parent.name
    if not (dst / 'latest.pt').exists():
        shutil.copytree(prev.parent, dst, dirs_exist_ok=True); print('resume from', prev)


def done(sensor):
    h = RUNS / f'retrain_{sensor}_s42/history.json'
    return h.exists() and json.loads(h.read_text())['epochs'][-1]['epoch'] >= EPOCHS[sensor]


procs = {}
for gpu, sensor in enumerate(('QB', 'GF2')):
    if done(sensor): continue
    run = RUNS / f'retrain_{sensor}_s42'; run.mkdir(exist_ok=True)
    resume = ' --resume' if (run / 'latest.pt').exists() else ''
    cmd = (f'{sys.executable} {BENCH}/train.py --sensor {sensor} --train {DATA}/train_{STEM[sensor]}.h5 --val {DATA}/valid_{STEM[sensor]}.h5 '
           f'--device cuda --workers 2 --out {run} --deadline {DEADLINE}{resume}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % max(1, torch.cuda.device_count())))
    print('$', cmd, f'(GPU {env["CUDA_VISIBLE_DEVICES"]})', flush=True)
    procs[sensor] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(run / 'train.log', 'a'), stderr=subprocess.STDOUT)
while procs:
    time.sleep(300)
    for sensor, p in list(procs.items()):
        tail = (RUNS / f'retrain_{sensor}_s42/train.log').read_text().strip().splitlines()[-1:]
        print(f'{(time.time() - T0) / 3600:.1f} h', sensor, tail, flush=True)
        if p.poll() is not None:
            print(sensor, 'exit', p.returncode, flush=True); procs.pop(sensor)

# %%
# 3. finished retrain runs: best.pt into the benchmark folder, RR + FR metrics, RR predictions
for sensor in ('QB', 'GF2'):
    run = RUNS / f'retrain_{sensor}_s42'
    if not done(sensor):
        print(sensor, 'not finished yet; attach this output as input next session'); continue
    ck = BENCH / f'checkpoints/retrain_{sensor}_s42'; ck.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / 'best.pt', ck / 'best.pt')
    res = BENCH / f'results/retrain_{sensor}_s42'; res.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / 'history.json', res / 'history.json')
    if not (res / 'metrics.json').exists():
        sh(f'{EVAL} --setting retrain --sensor {sensor} --seed 42')
        PRED[f'retrain_{sensor}'] = export_rr('retrain', sensor, 42)
(OUT / 'predictions.json').write_text(json.dumps(PRED, indent=1) + '\n')

# %%
# collect: benchmark folder results to bring back into the repository, plus run folders for the next session
sh(f'{EVAL} --summary', check=False)
for sub in ('results', 'checkpoints'):
    if (BENCH / sub).exists(): shutil.copytree(BENCH / sub, OUT / sub, dirs_exist_ok=True)
env = dict(ctrs_commit=CTRS_COMMIT, adwm_commit=UP_COMMIT, torch=torch.__version__, cuda=torch.version.cuda,
           gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], python=sys.version.split()[0],
           hours=round((time.time() - T0) / 3600, 2))
(OUT / 'environment.json').write_text(json.dumps(env, indent=1) + '\n')
print(json.dumps(env, indent=1)); sh(f'ls -la {OUT}')
