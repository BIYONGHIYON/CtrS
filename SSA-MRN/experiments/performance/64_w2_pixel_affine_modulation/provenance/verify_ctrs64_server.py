from pathlib import Path
import json, hashlib, ast, sys
base = Path('C:/CtrS-p64')
records = json.loads((base / 'ctrs64-source-manifest.json').read_text())
for rel, rec in records.items():
    p = base / rel
    assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rel
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'))
x = base / 'SSA-MRN/experiments/performance/64_w2_pixel_affine_modulation'
sys.path.insert(0, str(x))
import train, run, torch
names = set(train.T.train.__code__.co_names)
assert 'lambda' in str([getattr(k,'co_name','') for k in train.T.preflight.__code__.co_consts]), 'preflight patch not applied'
cfg = json.loads((x / 'config.json').read_text(encoding='utf8'))
from model import Model
for key in ('controls', 'reference_controls'):
    ck = torch.load(base / 'SSA-MRN' / cfg[key]['QB'], map_location='cpu', weights_only=False)
    m = Model('C2MS', 4, 42).cuda().eval(); m.load_state_dict(ck['model'], strict=True)
    with torch.inference_mode(): y = m(torch.ones(1, 1, 64, 64, device='cuda'), torch.ones(1, 4, 64, 64, device='cuda'), torch.ones(1, 4, 16, 16, device='cuda'))
    assert torch.isfinite(y).all()
train.write(base / 'source_verification.json', dict(time_kst=train.now(), verified_files=len(records), sealed_hashes=run.source_hashes(), preflight_patch=True, torch=torch.__version__))
print('SOURCE / PATCH / CUDA CONTROLS PASS', len(records), flush=True)
am = Model('C2MS_AM', 4, 42).cuda().eval()
with torch.inference_mode(): y = am(torch.ones(1, 1, 64, 64, device='cuda'), torch.ones(1, 4, 64, 64, device='cuda'), torch.ones(1, 4, 16, 16, device='cuda'))
assert torch.isfinite(y).all(); print('C2MS_AM CUDA FORWARD PASS', flush=True)
