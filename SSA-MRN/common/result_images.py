"""Standard result images for experiment reports: current baseline vs candidate on six land-cover-diverse RR scenes.

Scenes (QB RR test, 0-based) were picked from a GT-only contact sheet by land cover before any model output was viewed:
0 sea and harbour breakwater, 1 pier, curved breakwater and trees, 3 marina and sports fields, 7 forest park,
14 hill park and curved housing, 17 hill vegetation and trails.

Two images per scene:
- full_scene_XX.png: the whole 256x256 scene at native size. Top: LR MS (nearest) | PAN | baseline | candidate | GT.
  Bottom: baseline and candidate error maps |prediction - GT| on a shared colour scale (99th percentile of the baseline error).
- compare_scene_XX.png: the 64x64 window (stride 16) with the largest error reduction (baseline error - candidate error),
  magnified 4x: GT with yellow box | baseline | candidate | GT | baseline error | candidate error (since 2026-10-22 the same
  baseline -> candidate -> GT order as the full-scene image; earlier images put GT second). The window is chosen in
  favour of the candidate; whole-scene differences are given by the scene PSNR in the title and in result_images.json.
Display: RGB bands [2,1,0], shared GT 1-99% stretch per scene.
Other sensors (experiment 54): pass rr_file, scale (DN normalization), bands and land_cover; the defaults are the QB values above.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import evaluation_protocol as protocol

SCENES = (0, 1, 3, 7, 14, 17)
LAND_COVER = {0: '바다·항구 방파제', 1: '부두·곡선 방파제·수목', 3: '요트 마리나·운동장', 7: '숲 공원', 14: '언덕 공원·곡선 주택가', 17: '산지 녹지·산책로'}
BANDS = [2, 1, 0]
WINDOW, STRIDE, ZOOM = 64, 16, 4
FONTS = ('C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/AppleSDGothicNeo.ttc', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf')


def _font(size=15):
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rgb(x, lo, hi, bands=BANDS):
    return Image.fromarray((np.clip((x[bands].transpose(1, 2, 0) - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype('uint8'))


def _heat(m, vmax):
    v = np.clip(m / max(vmax, 1e-8), 0, 1)
    return Image.fromarray((np.stack([np.clip(1.5 * v, 0, 1), np.clip(1.5 * v - .5, 0, 1), np.clip(3 * v - 2, 0, 1)], -1) * 255).astype('uint8'))


def _psnr(pred, gt):
    return protocol.rr_metrics(gt, pred)['PSNR']


def _full(a, out, names, title, path, font, labels, bands=BANDS):
    base, cand = names
    lo, hi = np.percentile(a['gt'][bands].transpose(1, 2, 0), [1, 99], axis=(0, 1))
    pan = a['pan'][0]; pl, ph = np.percentile(pan, [1, 99])
    pan_img = Image.fromarray((np.clip((pan - pl) / max(ph - pl, 1e-8), 0, 1) * 255).astype('uint8')).convert('RGB')
    err = {k: np.abs(o - a['gt']).mean(0) for k, o in out.items()}; vmax = np.percentile(err[base], 99)
    size = a['gt'].shape[-1]
    rows = ([('LR MS (최근접 확대)', _rgb(a['ms'], lo, hi, bands).resize((size, size), Image.Resampling.NEAREST)), ('PAN 입력', pan_img),
             (labels[0], _rgb(out[base], lo, hi, bands)), (labels[1], _rgb(out[cand], lo, hi, bands)), ('정답 GT', _rgb(a['gt'], lo, hi, bands))],
            [None, None, (f'{labels[0]} 오차', _heat(err[base], vmax)), (f'{labels[1]} 오차', _heat(err[cand], vmax)), None])
    img = Image.new('RGB', (5 * (size + 8) + 8, 2 * (size + 30) + 36), 'white'); d = ImageDraw.Draw(img)
    d.text((10, 6), title, fill='black', font=font)
    for r, cells in enumerate(rows):
        for k, cell in enumerate(cells):
            if cell is None: continue
            x, y = 8 + k * (size + 8), 36 + r * (size + 30)
            d.text((x, y), cell[0], fill='black', font=font); img.paste(cell[1], (x, y + 24))
    img.save(path)


def _zoom(a, out, names, title, path, font, labels, bands=BANDS):
    base, cand = names
    size = a['gt'].shape[-1]
    gain = np.abs(out[base] - a['gt']).mean(0) - np.abs(out[cand] - a['gt']).mean(0)
    starts = range(0, size - WINDOW + 1, STRIDE)
    r, c = max(((r, c) for r in starts for c in starts), key=lambda rc: gain[rc[0]:rc[0] + WINDOW, rc[1]:rc[1] + WINDOW].mean())
    s = (slice(None), slice(r, r + WINDOW), slice(c, c + WINDOW))
    lo, hi = np.percentile(a['gt'][bands].transpose(1, 2, 0), [1, 99], axis=(0, 1))
    err = {k: np.abs(o - a['gt']).mean(0)[r:r + WINDOW, c:c + WINDOW] for k, o in out.items()}; vmax = np.percentile(err[base], 99)
    full = _rgb(a['gt'], lo, hi, bands); ImageDraw.Draw(full).rectangle([c, r, c + WINDOW - 1, r + WINDOW - 1], outline=(255, 210, 0), width=2)
    crop = {k: _psnr(o[s], a['gt'][s]) for k, o in out.items()}
    cells = [('정답 전체 (노란 상자)', full), (labels[0], _rgb(out[base][s], lo, hi, bands)), (labels[1], _rgb(out[cand][s], lo, hi, bands)),
             ('정답 (확대)', _rgb(a['gt'][s], lo, hi, bands)), (f'{labels[0]} 오차', _heat(err[base], vmax)), (f'{labels[1]} 오차', _heat(err[cand], vmax))]
    P = WINDOW * ZOOM; img = Image.new('RGB', (len(cells) * (P + 8) + 8, P + 64), 'white'); d = ImageDraw.Draw(img)
    d.text((10, 6), f'{title} · 확대 영역({cand} 개선이 가장 큰 64×64) PSNR {base} {crop[base]:.2f} → {cand} {crop[cand]:.2f} dB', fill='black', font=font)
    for k, (label, im) in enumerate(cells):
        x = 8 + k * (P + 8); d.text((x, 30), label, fill='black', font=font); img.paste(im.resize((P, P), Image.Resampling.NEAREST), (x, 56))
    img.save(path)
    return dict(window=[r, c], crop_psnr=crop, crop_mean_abs_err={k: float(v.mean()) for k, v in err.items()})


def _predict(net, a, scale, device):
    if scale == 2047:
        return protocol.predict(net, a['pan'], a['lms'], a['ms'], device)
    x = [torch.from_numpy((a[k] / scale).astype('float32')).unsqueeze(0).to(device) for k in ('pan', 'lms', 'ms')]
    with torch.no_grad():
        out = net(*x)[0].float().cpu().numpy().astype(np.float64) * scale
    if not np.isfinite(out).all():
        raise FloatingPointError('nonfinite prediction')
    return out


def render(baseline, candidate, data_dir, output_dir, names=('F1', 'candidate'), seed=42, device='cuda', scenes=SCENES, labels=None,
           rr_file='test_qb_multiExm1.h5', scale=2047., bands=BANDS, land_cover=LAND_COVER):
    """baseline/candidate: networks taking (pan, lms, ms) in DN/scale. Writes both images per scene and result_images.json.
    names are short names for titles; labels are panel labels (default: '현재 기준 <baseline>', '<candidate>')."""
    labels = labels or (f'현재 기준 {names[0]}', names[1])
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    font, stats = _font(), {}
    nets = dict(zip(names, (baseline, candidate)))
    with h5py.File(next(Path(data_dir).rglob(rr_file)), 'r') as f:
        for i in scenes:
            a = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
            out = {k: _predict(n, a, scale, device) for k, n in nets.items()}
            scene = {k: _psnr(o, a['gt']) for k, o in out.items()}
            title = f'RR 장면 {i}' + (f' ({land_cover[i]})' if i in land_cover else '') + f' · seed {seed} · 장면 PSNR {names[0]} {scene[names[0]]:.2f} → {names[1]} {scene[names[1]]:.2f} dB'
            _full(a, out, names, title + ' · 원본 크기, 오차 지도는 같은 색 척도', out_dir / f'full_scene_{i:02}.png', font, labels, bands)
            stats[i] = dict(land_cover=land_cover.get(i), scene_psnr=scene, **_zoom(a, out, names, title, out_dir / f'compare_scene_{i:02}.png', font, labels, bands))
    (out_dir / 'result_images.json').write_text(json.dumps(dict(names=list(names), seed=seed, scenes=list(scenes), stats=stats), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return stats
