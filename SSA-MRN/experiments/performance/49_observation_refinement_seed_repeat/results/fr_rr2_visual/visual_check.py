"""Visual check (no training): FR and RR2 outputs of C0, F1, C2, C3 (seed 42) on fixed scenes.

Scenes 1, 8, 11, 13, 19 (0-based) and the centre crop are fixed before looking at any output.
FR: PAN | MS (nearest) | C0 | F1 | C2 | C3 RGB, then |output - LMS| (injected detail) maps.
RR2: GT (RR MS) | degraded MS (nearest) | C0 | F1 | C2 | C3 RGB, then |output - GT| error maps.
"""
import sys, json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1])          # SSA-MRN root of the staged tree
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa: E402
sys.path.insert(0, str(ROOT / 'experiments/performance/48_parallel_capacity_observation_refinement'))
import model as M48  # noqa: E402

SCENES = (1, 8, 11, 13, 19)
BANDS = [2, 1, 0]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
E43 = ROOT / 'experiments/performance/43_normalized_parallel_capacity/checkpoints'
CKPT = {'C0': E43 / 'C0_QB_s42/best.pt', 'C2': E43 / 'C2_QB_s42/best.pt', 'C3': E43 / 'C3_QB_s42/best.pt',
        'F1': ROOT / 'experiments/performance/49_observation_refinement_seed_repeat/checkpoints/F1_QB_s42/best.pt'}
M43 = M48.M43


def build(arm):
    net = M43.Model(arm, 42) if arm in ('C0', 'C2', 'C3') else M48.Model('F1', 42)
    net.load_state_dict(torch.load(CKPT[arm], map_location='cpu', weights_only=False)['model'], strict=True)
    return net.to(DEV).eval()


def rgb(x, lo, hi):
    y = np.clip((x[BANDS].transpose(1, 2, 0) - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    return Image.fromarray((y * 255).astype('uint8'))


def heat(m, vmax):
    v = np.clip(m / max(vmax, 1e-8), 0, 1)
    r = np.clip(1.5 * v, 0, 1); g = np.clip(1.5 * v - 0.5, 0, 1); b = np.clip(3 * v - 2, 0, 1)
    return Image.fromarray((np.stack([r, g, b], -1) * 255).astype('uint8'))


def nearest(ms, size):
    f = size // ms.shape[-1]
    return ms.repeat(f, -2).repeat(f, -1)


def panel(cells, title, path, zoom):
    w, h = cells[0][1].size
    W, H = w * zoom, h * zoom
    cols = len(cells) // 2
    img = Image.new('RGB', (cols * W, 2 * (H + 22) + 24), 'white'); d = ImageDraw.Draw(img)
    d.text((6, 4), title, fill='black')
    for k, (label, im) in enumerate(cells):
        r, c = divmod(k, cols)
        x, y = c * W, 24 + r * (H + 22)
        d.text((x + 6, y + 4), label, fill='black')
        img.paste(im.resize((W, H), Image.Resampling.NEAREST), (x, y + 22))
    img.save(path)


nets = {arm: build(arm) for arm in ('C0', 'F1', 'C2', 'C3')}
degrade = protocol.WaldDegradation('QB', DEV)
stats = {}
with h5py.File(next(DATA.rglob('test_qb_OrigScale_multiExm1.h5')), 'r') as fr, h5py.File(next(DATA.rglob('test_qb_multiExm1.h5')), 'r') as rr:
    for i in SCENES:
        # FR, centre 128x128 crop of the 512x512 output
        a = {k: np.asarray(fr[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')}
        outs = {arm: protocol.predict(n, a['pan'], a['lms'], a['ms'], DEV) for arm, n in nets.items()}
        c = slice(192, 320)
        lo, hi = np.percentile(a['lms'][BANDS][:, c, c].transpose(1, 2, 0), [1, 99], axis=(0, 1))
        p = a['pan'][0][c, c]; pl, ph = np.percentile(p, [1, 99])
        pan_img = Image.fromarray((np.clip((p - pl) / max(ph - pl, 1e-8), 0, 1) * 255).astype('uint8')).convert('RGB')
        det = {arm: np.abs(o - a['lms']).mean(0)[c, c] for arm, o in outs.items()}
        vmax = max(np.percentile(v, 99) for v in det.values())
        cells = [('PAN', pan_img), ('MS (nearest)', rgb(nearest(a['ms'], 512)[:, c, c], lo, hi))] + [(arm, rgb(o[:, c, c], lo, hi)) for arm, o in outs.items()]
        cells += [('', Image.new('RGB', pan_img.size, 'white')), ('|out - LMS| scale %.0f DN' % vmax, Image.new('RGB', pan_img.size, 'white'))] + [(f'{arm} injected detail', heat(det[arm], vmax)) for arm in outs]
        panel(cells, f'FR scene {i} (no ground truth), centre crop 128x128 of 512x512, seed 42', OUT / f'FR_scene_{i:02}.png', 3)
        stats[f'FR_{i}'] = {arm: {'mean_abs_detail_DN': float(np.abs(o - a['lms']).mean())} for arm, o in outs.items()}
        # RR2: whole 64x64 output
        b = {k: np.asarray(rr[k][i], dtype=np.float64) for k in ('pan', 'ms')}
        pan_b, lms_b, ms_b = degrade(b['pan'], b['ms'])
        gt = b['ms']
        outs2 = {arm: protocol.predict(n, pan_b, lms_b, ms_b, DEV) for arm, n in nets.items()}
        lo, hi = np.percentile(gt[BANDS].transpose(1, 2, 0), [1, 99], axis=(0, 1))
        err = {arm: np.abs(o - gt).mean(0) for arm, o in outs2.items()}
        vmax = max(np.percentile(v, 99) for v in err.values())
        cells = [('GT (RR MS)', rgb(gt, lo, hi)), ('MS input (nearest)', rgb(nearest(ms_b, 64), lo, hi))] + [(arm, rgb(o, lo, hi)) for arm, o in outs2.items()]
        cells += [('', Image.new('RGB', (64, 64), 'white')), ('|out - GT| scale %.0f DN' % vmax, Image.new('RGB', (64, 64), 'white'))] + [(f'{arm} error', heat(err[arm], vmax)) for arm in outs2]
        panel(cells, f'RR2 scene {i} (GT available), full 64x64, seed 42', OUT / f'RR2_scene_{i:02}.png', 6)
        stats[f'RR2_{i}'] = {arm: {'PSNR': protocol.rr_metrics(gt, o)['PSNR'], 'mean_abs_err_DN': float(err[arm].mean())} for arm, o in outs2.items()}
        print(i, 'RR2 PSNR', {arm: round(v['PSNR'], 2) for arm, v in stats[f'RR2_{i}'].items()}, flush=True)
(OUT / 'stats.json').write_text(json.dumps(stats, indent=2))
