"""D3 (training-free, post-hoc): the sensor-level band calibration applied to other archived WV3 seed-42 models on WV2 (300-epoch C2MS of 59, blur-augmented arms of 67/71)."""
import sys, json, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
from common.metrics import rr_metrics
torch.set_num_threads(4); torch.set_grad_enabled(False)
RS = json.loads(Path(sys.argv[2]).read_text()); S = 2047.
MODELS = {'C2MS 150ep (54)': PERF / '54_multi_sensor_scale_mixed_training/checkpoints/C2MS_WV3_s42/best.pt',
          'C2MS 300ep (59)': PERF / '59_extended_cosine_training/checkpoints/C2MS_WV3_s42/best.pt',
          'C2MS_B blur aug (67)': PERF / '67_pan_brightness_blur_augmentation/checkpoints/C2MS_B_WV3_s42/best.pt',
          'L1S_GB gain+blur aug (67)': PERF / '67_pan_brightness_blur_augmentation/checkpoints/L1S_GB_WV3_s42/best.pt',
          'BLUR_COARSE (71)': PERF / '71_coarse_pair_pan_blur/checkpoints/WV3_BLUR_COARSE_s42/best.pt'}
def detail(y, gt):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    hi = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) * f)); hy, hg = hi(y), hi(gt)
    return float(np.sqrt((hy ** 2).mean() / (hg ** 2).mean())), float((hy * hg).sum() / np.sqrt((hy ** 2).sum() * (hg ** 2).sum()))
res = {}
for target in ('WV2', 'WV3'):
    p66.use(target); tag = p66.SENSORS[target]['tag']
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        data = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
    t_const = np.array(RS['WV3_test']['scene_ratio_mean']) / np.array(RS[f'{target}_test']['scene_ratio_mean'])
    for name, path in MODELS.items():
        if not path.exists(): print('missing', path); continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = p66.Model('C2MS', 8, int(ck['seed'])); m.load_state_dict(ck['model'], strict=True); m.eval()
        for cond, t in (('raw', np.ones(8)),) + ((('const', t_const),) if target == 'WV2' else ()):
            for proj in (False, True):
                rows = []; t0 = time.time(); tb = t[:, None, None]
                for x in data:
                    tin = [torch.from_numpy((a / S).astype('float32'))[None] for a in (x['pan'], tb * x['lms'], tb * x['ms'])]
                    y = m(*tin).double()[0].numpy() * S / tb
                    if proj: yy, _ = p66.project(torch.from_numpy(y / S)[None], torch.from_numpy(x['ms'] / S)[None], max_iter=10); y = yy[0].numpy() * S
                    r = rr_metrics(x['gt'], y); de, dc = detail(y, x['gt']); rows.append([float(r['PSNR']), float(r['SAM']), float(r['ERGAS']), de, dc])
                mean = np.mean(rows, 0); key = f'{target}|{name}|{cond}|{"P" if proj else "-"}'; res[key] = mean.tolist()
                print(f'{key:55s} PSNR {mean[0]:.3f} SAM {mean[1]:.3f} ERGAS {mean[2]:.3f} detail {mean[3]:.3f} corr {mean[4]:.3f}  {time.time() - t0:.0f}s', flush=True)
                Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
