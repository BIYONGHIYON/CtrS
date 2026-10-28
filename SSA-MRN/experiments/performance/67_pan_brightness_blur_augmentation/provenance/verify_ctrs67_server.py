"""Server check of the 67 package (CPU only; the GPU may be training): file hashes, syntax, trainer patch, augmentation checks, control/reference strict load."""
from pathlib import Path
import json, hashlib, ast, sys
base = Path('C:/CtrS-p67')
records = json.loads((base / 'ctrs67-source-manifest.json').read_text())
for rel, rec in records.items():
    p = base / rel
    assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rel
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'))
x = base / 'SSA-MRN/experiments/performance/67_pan_brightness_blur_augmentation'
sys.path.insert(0, str(x))
import train, run, torch
names = set(train.T.train.__code__.co_names)
assert {'recon', 'augment_pan', 'spectral_loss'} <= names and 'square' not in names, 'trainer patch not applied'
cfg = json.loads((x / 'config.json').read_text(encoding='utf8'))
assert cfg['sensors'] == ['WV3'] and cfg['seeds'] == [42] and [q[1] for q in cfg['queue']] == ['L1S_GB', 'C2MS_GB', 'L1S_G', 'L1S_B', 'C2MS_G', 'C2MS_B']
checks = train.augment_check(cfg)
from model import Model
for rel in [cfg['controls']['WV3'], *cfg['reference_controls']['WV3'].values()]:
    ck = torch.load(base / 'SSA-MRN' / rel.replace('<seed>', '42'), map_location='cpu', weights_only=False)
    assert ck['seed'] == 42 and ck['sensor'] == 'WV3', rel
    m = Model('C2MS', 8, 42).eval(); m.load_state_dict(ck['model'], strict=True)
    with torch.inference_mode(): y = m(torch.ones(1, 1, 64, 64), torch.ones(1, 8, 64, 64), torch.ones(1, 8, 16, 16))
    assert torch.isfinite(y).all()
train.write(base / 'source_verification.json', dict(time_kst=train.now(), verified_files=len(records), sealed_hashes=run.source_hashes(), trainer_patch=True, augment_checks=checks, controls_strict_load='cpu', torch=torch.__version__))
print('SOURCE / TRAINER PATCH / AUGMENTATION / CONTROLS PASS', len(records), flush=True)
