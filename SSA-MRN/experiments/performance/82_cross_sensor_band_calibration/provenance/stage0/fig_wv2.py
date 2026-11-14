"""Visual check (post-hoc): WV3-trained models on WV2, with and without the sensor-level band calibration. Fixed scenes, fixed central crop (not cherry-picked)."""
import sys, json, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'; plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
from common.metrics import rr_metrics
torch.set_num_threads(2); torch.set_grad_enabled(False)
RS = json.loads(Path(sys.argv[1]).read_text()); S = 2047.; seed = int(sys.argv[3]); scenes = [int(s) for s in sys.argv[4].split(',')]
p66.use('WV2')
with h5py.File(ROOT / 'data/test/test_wv2_multiExm1.h5') as f:
    data = {i: {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in scenes}
t_const = np.array(RS['WV3_test']['scene_ratio_mean']) / np.array(RS['WV2_test']['scene_ratio_mean'])     # one constant per band (input-only statistics)
M = {n: p66.load_model(p, 8) for n, p in p66.models(seed, 'WV3').items()}
def run(name, x, t, proj):
    tb = t[:, None, None]; tin = [torch.from_numpy((a / S).astype('float32'))[None] for a in (x['pan'], tb * x['lms'], tb * x['ms'])]
    y = M[name](*tin).double()[0].numpy() * S / tb
    if proj: yy, _ = p66.project(torch.from_numpy(y / S)[None], torch.from_numpy(x['ms'] / S)[None], max_iter=10); y = yy[0].numpy() * S
    return y
def high_energy(y, gt):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    hi = lambda a: np.real(np.fft.ifft2(np.fft.fft2(a) * f)); return float(np.sqrt((hi(y) ** 2).mean() / (hi(gt) ** 2).mean()))
one = np.ones(8); RGB = [4, 2, 1]
cols = [('PAN 입력', None), ('C2MS (기준, 보정 없음)', ('C2MS', one, False)), ('L1·분광 + 사영\n(보정 없음)', ('L1S', one, True)),
        ('L1·분광 + 사영\n(밴드 보정)', ('L1S', t_const, True)), ('C2MS + 사영\n(밴드 보정)', ('C2MS', t_const, True)), ('정답', None)]
fig, ax = plt.subplots(2 * len(scenes), len(cols), figsize=(3.1 * len(cols), 3.3 * 2 * len(scenes)))
for si, i in enumerate(scenes):
    x = data[i]; gt = x['gt']; lo = np.percentile(gt[RGB], 1, axis=(1, 2))[:, None, None]; hi = np.percentile(gt[RGB], 99, axis=(1, 2))[:, None, None]
    show = lambda a: np.clip((a[RGB] - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)
    c0, c1 = 80, 176                                                              # fixed central 96x96 crop
    for ci, (title, spec_) in enumerate(cols):
        if title.startswith('PAN'): img = np.clip((x['pan'][0] - np.percentile(x['pan'], 1)) / (np.percentile(x['pan'], 99) - np.percentile(x['pan'], 1)), 0, 1); full = np.stack([img] * 3, -1); lab = title
        elif title == '정답': full = show(gt); lab = title
        else:
            y = run(*spec_[:1], x, *spec_[1:]); r = rr_metrics(gt, y); full = show(y)
            lab = f"{title}\nPSNR {r['PSNR']:.2f} dB · SAM {r['SAM']:.2f}° · 세부 {high_energy(y, gt):.2f}"
        ax[2 * si, ci].imshow(full, interpolation='nearest'); ax[2 * si, ci].set_title((f'장면 {i} · ' if ci == 0 else '') + lab, fontsize=8.5)
        ax[2 * si, ci].add_patch(plt.Rectangle((c0, c0), c1 - c0, c1 - c0, fill=False, ec='yellow', lw=1))
        ax[2 * si + 1, ci].imshow(full[c0:c1, c0:c1], interpolation='nearest'); ax[2 * si + 1, ci].set_title('가운데 96×96 확대', fontsize=8)
        ax[2 * si, ci].axis('off'); ax[2 * si + 1, ci].axis('off')
fig.suptitle(f'WV3 학습 모델(seed {seed})을 WV2에 적용: 밴드별 센서 보정 전후 (RGB 밴드 [4,2,1], 정답 1-99 퍼센타일 대비, 세부 = 8 px 미만 대역 세기 ÷ 정답)', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(sys.argv[2], dpi=110); print('saved', sys.argv[2])
