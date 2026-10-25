from pathlib import Path
import json, hashlib, ast, sys
base = Path('C:/CtrS-p61')
records = json.loads((base / 'ctrs61-source-manifest.json').read_text())
for rel, rec in records.items():
    p = base / rel
    assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rel
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'))
x = base / 'SSA-MRN/experiments/performance/61_high_frequency_band_ratio_correction'
sys.path.insert(0, str(x))
import train, torch
cfg = json.loads((x / 'config.json').read_text(encoding='utf8'))['phase1']
m = train.c2ms_model(cfg, 'QB', 42)
with torch.inference_mode(): y = m(torch.ones(1, 1, 64, 64, device='cuda'), torch.ones(1, 4, 64, 64, device='cuda'), torch.ones(1, 4, 16, 16, device='cuda'))
assert y.shape == (1, 4, 64, 64) and torch.isfinite(y).all()
train.write(base / 'source_verification.json', dict(time_kst=train.now(), verified_files=len(records), torch=torch.__version__, strict_load_c2ms_qb_s42=True))
print('SOURCE / IMPORT / CUDA CONTROL PASS', len(records), flush=True)
