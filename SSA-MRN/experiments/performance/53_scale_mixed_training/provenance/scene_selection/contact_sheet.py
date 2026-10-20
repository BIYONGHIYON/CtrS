"""GT-only contact sheet of the 20 QB RR test scenes (no model output), used to pick scenes by land cover."""
import sys
from pathlib import Path
import h5py, numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FONT = ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 18)
B = [2, 1, 0]
with h5py.File(next(DATA.rglob('test_qb_multiExm1.h5')), 'r') as f:
    gts = [np.asarray(f['gt'][i], dtype=np.float64) for i in range(len(f['gt']))]
S = 256; sheet = Image.new('RGB', (5 * (S + 6) + 6, 4 * (S + 6) + 6), 'white'); d = ImageDraw.Draw(sheet)
for i, g in enumerate(gts):
    x = g[B].transpose(1, 2, 0); lo, hi = np.percentile(x, [1, 99], axis=(0, 1))
    im = Image.fromarray((np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype('uint8'))
    r, c = divmod(i, 5); px, py = 6 + c * (S + 6), 6 + r * (S + 6)
    sheet.paste(im, (px, py)); d.rectangle([px, py, px + 34, py + 26], fill='black'); d.text((px + 4, py + 2), str(i), fill='yellow', font=FONT)
sheet.save(OUT)
