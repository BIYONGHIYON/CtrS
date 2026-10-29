"""Server check of the 69 package (CPU only): file hashes, syntax, trainer patch, loss forms, control/reference strict load."""
from pathlib import Path
import json, hashlib, ast, sys
base = Path('C:/CtrS-p69')
records = json.loads((base / 'ctrs69-source-manifest.json').read_text())
for rel, rec in records.items():
    p = base / rel
    assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rel
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'))
x = base / 'SSA-MRN/experiments/performance/69_low_frequency_squared_error_loss'
sys.path.insert(0, str(x))
import train, run, torch
names = set(train.T.train.__code__.co_names)
assert {'recon', 'phase_table', 'spectral_loss'} <= names and 'square' not in names, 'trainer patch not applied'
cfg = json.loads((x / 'config.json').read_text(encoding='utf8'))
assert cfg['sensors'] == ['QB', 'WV3'] and cfg['seeds'] == [42] and [q[:2] for q in cfg['queue']] == [['QB', 'C2MS_L1SO'], ['QB', 'C2MS_SPLIT'], ['WV3', 'C2MS_L1SO'], ['WV3', 'C2MS_SPLIT']]
assert cfg['loss_weights']['lambda_O'] == 500.0 and cfg['loss_weights']['k_S'] == 431.5
checks = train.loss_check({**cfg, 'sensors': []})  # loss forms; the phase tables are checked in preflight after prepare
from model import Model
bands = {'QB': 4, 'WV3': 8}
for sensor in cfg['sensors']:
    for rel in [cfg['controls'][sensor], *cfg['reference_controls'][sensor].values()]:
        ck = torch.load(base / 'SSA-MRN' / rel.replace('<seed>', '42'), map_location='cpu', weights_only=False)
        assert ck['seed'] == 42 and ck.get('sensor', 'QB') == sensor, rel
        m = Model('C2MS', bands[sensor], 42).eval(); m.load_state_dict(ck['model'], strict=True)
        with torch.inference_mode(): y = m(torch.ones(1, 1, 64, 64), torch.ones(1, bands[sensor], 64, 64), torch.ones(1, bands[sensor], 16, 16))
        assert torch.isfinite(y).all()
train.write(base / 'source_verification.json', dict(time_kst=train.now(), verified_files=len(records), sealed_hashes=run.source_hashes(), trainer_patch=True, loss_forms=checks, controls_strict_load='cpu', torch=torch.__version__))
print('SOURCE / TRAINER PATCH / LOSS FORMS / CONTROLS PASS', len(records), flush=True)
