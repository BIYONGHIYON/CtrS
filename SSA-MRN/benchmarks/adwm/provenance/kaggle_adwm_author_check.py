# Kaggle notebook: ADWM author-weight reproduction check (CtrS), percent format. No training.
# Inputs: dataset ctrs-pan-sharpening-dataset; accelerator GPU T4 (one is enough); internet on.
#
# The published WV3 weights load into the official BWNet only without the covariance LayerNorm (14 tensors missing), and the
# first run of setting 'author' gave WV3 RR PSNR 37.858 against the paper's 39.170. This notebook evaluates every reading of
# the published files on WV3 RR test (20 scenes, CtrS common/metrics) so that the reading that reproduces the paper can be
# identified and all of them reported:
#   file   : weights/BWNET_LAGConv_WV3/model.pth (named for this model)
#            weights/BWNET_FusionNet_WV3/model.pth (replaced in the last upstream commit; same keys as the LAGConv model)
#   reading: author    = covariance LayerNorm removed (identity)
#            author_ln = parameter-free LayerNorm (equal to the official LayerNorm at its initial values)

# %%
import json, subprocess, sys, time, hashlib
from pathlib import Path
T0 = time.time(); TMP = Path('/tmp/ctrs'); TMP.mkdir(exist_ok=True); OUT = Path('/kaggle/working/adwm_author_check'); OUT.mkdir(exist_ok=True)
def sh(cmd, cwd=None):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode: raise RuntimeError(cmd)
    return r.stdout
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
sh('pip install -q einops')
CTRS = TMP / 'CtrS'
if not CTRS.exists():
    sh(f'git clone -q --depth 1 --branch adwm-benchmark --filter=blob:none --sparse https://github.com/BIYONGHIYON/CtrS.git {CTRS}')
    sh('git sparse-checkout set SSA-MRN/common SSA-MRN/docs/images/c2ms_sp SSA-MRN/benchmarks/adwm', cwd=CTRS)
    sh('git submodule update --init SSA-MRN/benchmarks/adwm/upstream', cwd=CTRS)
BENCH = CTRS / 'SSA-MRN/benchmarks/adwm'
print('CtrS', sh('git rev-parse HEAD', cwd=CTRS).strip(), 'ADWM', sh('git rev-parse HEAD', cwd=BENCH / 'upstream').strip())

# %%
import h5py, numpy as np, torch
sys.path.insert(0, str(CTRS / 'SSA-MRN'))
from common.benchmark_evaluate import load_model
from common.metrics import rr_metrics
rr = [p for p in Path('/kaggle/input').rglob('*.h5') if p.name == 'test_wv3_multiExm1.h5' or p.name.endswith('__test_wv3_multiExm1.h5')][0]
print(rr, sha256(rr))
with h5py.File(rr, 'r') as h: scenes = [{k: np.asarray(h[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(h['gt'].shape[0])]
FILES = {'LAGConv_file': BENCH / 'upstream/weights/BWNET_LAGConv_WV3/model.pth', 'FusionNet_file': BENCH / 'upstream/weights/BWNET_FusionNet_WV3/model.pth'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {}
for fname, path in FILES.items():
    for setting in ('author', 'author_ln'):
        net, _ = load_model(BENCH, setting, 'WV3', None, path, device)
        rows = []
        for s in scenes:
            x = [torch.from_numpy((s[k] / 2047.).astype('float32'))[None].to(device) for k in ('pan', 'lms', 'ms')]
            with torch.no_grad(): y = net(*x)[0].double().cpu().numpy() * 2047.
            rows.append(rr_metrics(s['gt'], y))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[f'{fname}/{setting}'] = dict(file=path.name if fname == 'LAGConv_file' else 'BWNET_FusionNet_WV3/model.pth', sha256=sha256(path),
                                             setting=setting, mean=mean, per_scene_psnr=[r['PSNR'] for r in rows])
        print(f'{fname:15s} {setting:10s}', {k: round(v, 4) for k, v in mean.items()}, flush=True)
print('paper (ADWM Table 1, WV3): PSNR 39.170  SAM 2.913  ERGAS 2.145  Q8 0.921')
(OUT / 'author_check.json').write_text(json.dumps(dict(test_file=rr.name, test_sha256=sha256(rr), device=device, torch=torch.__version__,
    paper=dict(PSNR=39.17, SAM=2.913, ERGAS=2.145, Q8=0.921), results=results), indent=1) + '\n')
print('elapsed', round(time.time() - T0), 's')
