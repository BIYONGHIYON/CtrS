"""Original SSA-MRN K4 (43 C0 seed 42) vs current baseline F1 (49 seed 42) on fixed RR scenes 1, 8, 11, 13, 19.
Zoom window: the 64x64 window (stride 16) with the largest original-model error; chosen without looking at F1."""
import sys, json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
ROOT, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa
sys.path.insert(0, str(ROOT / 'experiments/performance/48_parallel_capacity_observation_refinement'))
import model as M48  # noqa
FONT = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 15)
def load(arm):
    net = M48.M43.Model('C0', 42) if arm == 'C0' else M48.Model('F1', 42)
    p = ROOT / ('experiments/performance/43_normalized_parallel_capacity/checkpoints/C0_QB_s42/best.pt' if arm == 'C0' else 'experiments/performance/49_observation_refinement_seed_repeat/checkpoints/F1_QB_s42/best.pt')
    net.load_state_dict(torch.load(p, map_location='cpu', weights_only=False)['model'], strict=True); return net.eval()
nets = {'C0': load('C0'), 'F1': load('F1')}
B = [2, 1, 0]; W, Z = 64, 4; stats = {}
def rgb(x, lo, hi): return Image.fromarray((np.clip((x[B].transpose(1, 2, 0) - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype('uint8'))
def heat(m, vmax):
    v = np.clip(m / max(vmax, 1e-8), 0, 1); return Image.fromarray((np.stack([np.clip(1.5 * v, 0, 1), np.clip(1.5 * v - .5, 0, 1), np.clip(3 * v - 2, 0, 1)], -1) * 255).astype('uint8'))
psnr = lambda a, b: protocol.rr_metrics(b, a)['PSNR']
with h5py.File(next(DATA.rglob('test_qb_multiExm1.h5')), 'r') as f:
    for i in (1, 8, 11, 13, 19):
        a = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
        out = {k: protocol.predict(n, a['pan'], a['lms'], a['ms'], 'cpu') for k, n in nets.items()}
        e0 = np.abs(out['C0'] - a['gt']).mean(0)
        best = max(((r, c) for r in range(0, 256 - W + 1, 16) for c in range(0, 256 - W + 1, 16)), key=lambda rc: e0[rc[0]:rc[0] + W, rc[1]:rc[1] + W].mean())
        r, c = best; s = (slice(None), slice(r, r + W), slice(c, c + W))
        lo, hi = np.percentile(a['gt'][B].transpose(1, 2, 0), [1, 99], axis=(0, 1))
        err = {k: np.abs(o - a['gt']).mean(0)[r:r + W, c:c + W] for k, o in out.items()}
        vmax = np.percentile(err['C0'], 99)
        full = rgb(a['gt'], lo, hi); d = ImageDraw.Draw(full); d.rectangle([c, r, c + W - 1, r + W - 1], outline=(255, 210, 0), width=2)
        cells = [('정답 전체 (노란 상자 = 확대 영역)', full.resize((256, 256), Image.Resampling.NEAREST)), ('정답 (확대)', rgb(a['gt'][s], lo, hi)), ('원본 SSA-MRN', rgb(out['C0'][s], lo, hi)),
                 ('현재 기준 F1', rgb(out['F1'][s], lo, hi)), ('원본 오차 |예측-정답|', heat(err['C0'], vmax)), ('F1 오차 |예측-정답|', heat(err['F1'], vmax))]
        P = W * Z; img = Image.new('RGB', (len(cells) * (P + 8) + 8, P + 64), 'white'); d = ImageDraw.Draw(img)
        sc = {k: psnr(o, a['gt']) for k, o in out.items()}; cr = {k: psnr(o[s], a['gt'][s]) for k, o in out.items()}
        d.text((10, 6), f'RR 장면 {i} · seed 42 · 장면 PSNR 원본 {sc["C0"]:.2f} → F1 {sc["F1"]:.2f} dB · 확대 영역 PSNR 원본 {cr["C0"]:.2f} → F1 {cr["F1"]:.2f} dB', fill='black', font=FONT)
        for k, (label, im) in enumerate(cells):
            x = 8 + k * (P + 8); d.text((x, 30), label, fill='black', font=FONT); img.paste(im.resize((P, P), Image.Resampling.NEAREST), (x, 56))
        img.save(OUT / f'baseline_compare_scene_{i:02}.png')
        stats[i] = dict(window=[r, c], scene_psnr=sc, crop_psnr=cr, crop_mean_abs_err={k: float(v.mean()) for k, v in err.items()})
        print(i, stats[i], flush=True)
(OUT / 'baseline_compare.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
