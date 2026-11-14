"""Reference only (local CPU, after the evaluation): WV3-trained K2 (seeds 42-44) on the WV2 test with the band calibration of experiment 82
(prepare.Calibrated, mode 'scene', WV3 training-set ratios) + K10 projection, next to the same for 54 C2MS and 65 C2MS_L1S. Transfer tests are
not judged (criteria v2.4)."""
import importlib.util, json, sys
from pathlib import Path
import h5py, numpy as np, torch
HERE = Path(__file__).resolve().parents[1]; ROOT = HERE.parents[2]; sys.path.insert(0, str(ROOT))
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
M = load('m83', HERE / 'model.py'); P82 = load('p82', HERE.parent / '82_cross_sensor_band_calibration/prepare.py')
from common.metrics import rr_metrics
torch.set_grad_enabled(False); torch.set_num_threads(7)
ref = json.loads((HERE.parent / '82_cross_sensor_band_calibration/results/reference_stats/WV3.json').read_text())
with h5py.File(ROOT / 'data/test/test_wv2_multiExm1.h5') as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
P = 'experiments/performance/'
nets = {'C2MS': (P + '54_multi_sensor_scale_mixed_training/checkpoints/C2MS_WV3_s{}/best.pt', 'C2MS'), 'L1S': (P + '65_l1_spectral_loss/checkpoints/C2MS_L1S_WV3_s{}/best.pt', 'C2MS'),
        'K2': (P + '83_factorized_c2ms_finetuning/checkpoints/K2_WV3_s{}/best.pt', 'K2')}
res = {}
for seed in (42, 43, 44):
    for name, (path, kind) in nets.items():
        ck = torch.load(ROOT / path.format(seed), map_location='cpu', weights_only=False)
        m = M.M54.Model('C2MS', 8, seed) if kind == 'C2MS' else M.Model('K2', 8, seed); m.load_state_dict(ck['model'], strict=True); m.eval()
        for cond, net in (('raw+P', P82.Projected(m, 'WV2')), ('cal+P', P82.Projected(P82.Calibrated(m, ref, 'scene'), 'WV2'))):
            rows = []
            for a in rr:
                y = net(*[torch.from_numpy((a[k] / 2047.).astype('float32'))[None] for k in ('pan', 'lms', 'ms')])[0].double().numpy() * 2047
                r = rr_metrics(a['gt'], y); rows.append([float(r['PSNR']), float(r['SAM']), float(r['ERGAS'])])
            res[f'{name}|s{seed}|{cond}'] = np.mean(rows, 0).tolist(); print(name, seed, cond, np.round(res[f'{name}|s{seed}|{cond}'], 4), flush=True)
mean = {k: np.mean([res[f'{k.split("|")[0]}|s{s}|{k.split("|")[1]}'] for s in (42, 43, 44)], 0).tolist() for k in [f'{n}|{c}' for n in nets for c in ('raw+P', 'cal+P')]}
(HERE / 'results/wv2_band_calibration.json').write_text(json.dumps(dict(note=__doc__, per_seed=res, mean_3seed=mean), indent=1, ensure_ascii=False))
print(json.dumps({k: [round(x, 3) for x in v] for k, v in mean.items()}))
