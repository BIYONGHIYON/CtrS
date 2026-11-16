# Kaggle notebook ctrs-cannet-author-repro (version 1, 2026-11-17 KST), percent format: each "# %%" is one notebook cell.
# Import with File > Import Notebook. Input: dataset ctrs-pan-sharpening-dataset; accelerator GPU T4 x2; internet on. No training.

# %% [markdown]
# # CANNet author-weight reproduction (CtrS, no training)
#
# Official code Duanyll/CANConv @ `0b7cc6f4ed257fcdca806d1d4fe8ec2ebf4beee9`, published weights `weights/cannet_{qb,gf2,wv3}.pth`.
# 1. test-file SHA-256 vs the public PanCollection files (HuggingFace elsting/PanCollection main)
# 2. official inference path `canconv.scripts.test.run_test` (same as the authors), repeated 3 times per sensor (k-means is not deterministic)
# 3. metrics with CtrS `common/metrics` (DLPan definitions), compared with the paper numbers
# 4. CtrS adapter path `common/benchmark_evaluate.py --setting author` (RR + FR, metrics.json)
# Accelerator GPU T4 x2 (one GPU used), internet on. Every process sees ONE GPU (kmcuda needs peer access otherwise).

# %%
import os, sys, json, time, hashlib, shutil, subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
T0 = time.time(); W = Path('/kaggle/working'); TMP = Path('/tmp/ctrs'); TMP.mkdir(exist_ok=True)
def sh(cmd, cwd=None, env=None):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True, env=env)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode: raise RuntimeError(f'failed ({r.returncode}): {cmd}')
    return r.stdout
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'visible gpus', torch.cuda.device_count())

# %%
# official code + native operators
UP = TMP / 'CANConv'
if not (UP / 'canconv/layers/pwac/pwacnative.so').exists():
    if not UP.exists():
        sh(f'git clone -q https://github.com/Duanyll/CANConv.git {UP}')
        sh('git checkout -q 0b7cc6f4ed257fcdca806d1d4fe8ec2ebf4beee9 && git submodule update --init --recursive -q', cwd=UP)
    sh('pip install -q ninja cmake einops')
    cc = torch.cuda.get_device_capability(0); arch = f'{cc[0]}{cc[1]}'; py = sys.executable
    sh(f'cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH={arch} -DCMAKE_CUDA_ARCHITECTURES={arch} '
       f'-DPython3_EXECUTABLE={py} -DPYTHON_EXECUTABLE={py} -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -Bbuild -G Ninja', cwd=UP)
    sh('cmake --build build --config Release --target all', cwd=UP)
print(sh('git rev-parse HEAD && git -C kmcuda rev-parse HEAD', cwd=UP))
WEIGHTS = {k: dict(path=str(UP / f'weights/cannet_{k}.pth'), sha256=sha256(UP / f'weights/cannet_{k}.pth')) for k in ('qb', 'gf2', 'wv3')}
print(json.dumps(WEIGHTS, indent=1))
CTRS = TMP / 'CtrS'
if not CTRS.exists():
    sh(f'git clone -q --depth 1 --filter=blob:none --sparse https://github.com/BIYONGHIYON/CtrS.git {CTRS}')
    sh('git sparse-checkout set SSA-MRN/common SSA-MRN/docs/images/c2ms_sp', cwd=CTRS)
CTRS_COMMIT = sh('git rev-parse HEAD', cwd=CTRS).strip()
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
# test files: Kaggle dataset copy, SHA-256 vs public PanCollection (HF elsting/PanCollection main, LFS oid = SHA-256)
import urllib.request
DATA = TMP / 'data'; DATA.mkdir(exist_ok=True)
files = [p for p in Path('/kaggle/input').rglob('*.h5') if p.stat().st_size > 1 << 20]
names = [f'test_{s}{o}_multiExm1.h5' for s in ('qb', 'gf2', 'wv3', 'wv2') for o in ('', '_OrigScale')]
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
    DATA_HASH[n] = dict(source=hits[0].name, sha256=sha256(hits[0]), hf_main=HF.get(n))
    print(n, DATA_HASH[n]['sha256'][:16], 'HF main', (HF.get(n) or '-')[:16], 'same' if HF.get(n) == DATA_HASH[n]['sha256'] else 'DIFFERENT/unknown')

# %%
# official inference (canconv.scripts.test.run_test, same path as the authors) x3, scored with CtrS common/metrics
import h5py, numpy as np, scipy.io as sio
sys.path.insert(0, str(UP)); sys.path.insert(0, str(CTRS / 'SSA-MRN'))
from common.metrics import rr_metrics
PAPER = {'QB': dict(SAM=4.507, ERGAS=3.652, Q4=0.937, PSNR_other=38.488),   # CANNet Table 2; PSNR from S2WMamba Table 3
         'GF2': dict(SAM=0.707, ERGAS=0.630, Q4=0.983, PSNR_other=49.52),   # CANNet Table 3; PSNR from S2WMamba/FusionMamba
         'WV3': dict(SAM=2.930, ERGAS=2.158, Q8=0.920, PSNR_other=39.003)}  # CANNet Table 1; PSNR from S2WMamba/FusionMamba
CASES = [('QB', 'qb', 'qb'), ('GF2', 'gf2', 'gf2'), ('WV3', 'wv3', 'wv3'), ('WV2', 'wv2', 'wv3')]  # sensor, preset, weights
REPEATS = 3
REPRO = W / 'repro'; REPRO.mkdir(exist_ok=True)
summary = dict(code_commit='0b7cc6f4ed257fcdca806d1d4fe8ec2ebf4beee9', ctrs_commit=CTRS_COMMIT, weights=WEIGHTS, data=DATA_HASH, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), repeats=REPEATS, results={})
os.chdir(UP)  # run_test reads presets.json from the working directory
from canconv.scripts.test import run_test
for sensor, preset, w in CASES:
    test_file = DATA / f'test_{preset}_multiExm1.h5'
    with h5py.File(test_file, 'r') as h: gt = np.asarray(h['gt'], dtype=np.float64)
    scale = 1023. if sensor == 'GF2' else 2047.
    runs = []
    for r in range(REPEATS):
        out = REPRO / f'{sensor}_rr_run{r}'; t = time.time()
        run_test('cannet', WEIGHTS[w]['path'], str(test_file), preset=preset, output_dir=str(out))
        mats = sorted((out / 'results').glob('*.mat'), key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
        rows = []
        for i, m in enumerate(mats):
            x = sio.loadmat(m); key = [k for k in x if not k.startswith('__')][0]
            pred = np.asarray(x[key], dtype=np.float64).transpose(2, 0, 1)   # saved H x W x C in DN
            rows.append(rr_metrics(gt[i], pred))
        mean = {k: float(np.mean([q[k] for q in rows])) for k in rows[0]}
        runs.append(dict(mean=mean, scenes=len(rows), seconds=round(time.time() - t, 1)))
        print(sensor, 'run', r, {k: round(v, 4) for k, v in mean.items()}, f'{len(rows)} scenes', flush=True)
        if r: shutil.rmtree(out)  # keep the first run's predictions only
    keys = runs[0]['mean'].keys()
    agg = {k: dict(mean=float(np.mean([q['mean'][k] for q in runs])), min=float(min(q['mean'][k] for q in runs)), max=float(max(q['mean'][k] for q in runs))) for k in keys}
    summary['results'][sensor] = dict(weights=w, test_file=test_file.name, runs=runs, aggregate=agg, paper=PAPER.get(sensor))
    p = PAPER.get(sensor)
    if p:
        q = 'Q4' if 'Q4' in p else 'Q8'
        print(f"{sensor} mean of {REPEATS}: PSNR {agg['PSNR']['mean']:.3f} [{agg['PSNR']['min']:.3f}, {agg['PSNR']['max']:.3f}] (other papers {p['PSNR_other']}) | "
              f"SAM {agg['SAM']['mean']:.3f} (paper {p['SAM']}) | ERGAS {agg['ERGAS']['mean']:.3f} (paper {p['ERGAS']}) | Q2n {agg['Q2n']['mean']:.4f} (paper {q} {p[q]})", flush=True)
(REPRO / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
os.chdir(W)

# %%
# CtrS adapter path: RR + FR, metrics.json per sensor (setting author)
EVAL = f'{sys.executable} {CTRS}/SSA-MRN/common/benchmark_evaluate.py --folder {BENCH} --data {DATA} --device cuda --setting author'
for sensor, w in (('QB', 'qb'), ('GF2', 'gf2'), ('WV3', 'wv3'), ('WV2', 'wv3')):
    try: sh(f'{EVAL} --sensor {sensor} --checkpoint {WEIGHTS[w]["path"]}')
    except Exception as e: print('adapter eval failed', sensor, e)
for p in sorted((BENCH / 'results').glob('*/metrics.json')):
    m = json.loads(p.read_text()); print(p.parent.name, {k: round(v, 4) for k, v in m['rr']['mean'].items()}, m['fr']['mean'])
print('elapsed', round((time.time() - T0) / 60, 1), 'min')
