from pathlib import Path
import json, hashlib, ast, sys
base = Path('C:/CtrS-p65s3')
records = json.loads((base / 'ctrs65s3-source-manifest.json').read_text())
for rel, rec in records.items():
    p = base / rel
    assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rel
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'))
x = base / 'SSA-MRN/experiments/performance/65_l1_spectral_loss'
sys.path.insert(0, str(x))
import train, run, torch
names = set(train.T.train.__code__.co_names)
assert 'abs' in names and 'square' not in names and 'spectral_loss' in names, 'L1 + spectral patch not applied'
cfg = json.loads((x / 'config.json').read_text(encoding='utf8'))
assert abs(train.T.spectral_weight(cfg, 'C2MS_L1S') - 0.46) < 1e-12 and cfg['seeds'] == [42, 43, 44] and cfg['sensors'] == ['GF2', 'WV3']
assert cfg['queue'] == [[s, 'C2MS_L1S', d] for d in (42, 43, 44) for s in ('GF2', 'WV3')]
from model import Model
from prepare import SENSORS
for seed in cfg['seeds']:
    for sensor in cfg['sensors']:
        nb = SENSORS[sensor]['bands']
        for rel in [cfg['controls'][sensor], *cfg['reference_controls'][sensor].values()]:
            ck = torch.load(base / 'SSA-MRN' / rel.replace('<seed>', str(seed)), map_location='cpu', weights_only=False)
            assert ck['seed'] == seed and ck['sensor'] == sensor, rel
            m = Model('C2MS', nb, seed).cuda().eval(); m.load_state_dict(ck['model'], strict=True)
            with torch.inference_mode(): y = m(torch.ones(1, 1, 64, 64, device='cuda'), torch.ones(1, nb, 64, 64, device='cuda'), torch.ones(1, nb, 16, 16, device='cuda'))
            assert torch.isfinite(y).all()
train.write(base / 'source_verification.json', dict(time_kst=train.now(), verified_files=len(records), sealed_hashes=run.source_hashes(), l1_spectral_patch=True, seeds=cfg['seeds'], sensors=cfg['sensors'], torch=torch.__version__))
print('SOURCE / L1+SPECTRAL PATCH / SEEDED GF2-WV3 CONTROLS PASS', len(records), flush=True)
