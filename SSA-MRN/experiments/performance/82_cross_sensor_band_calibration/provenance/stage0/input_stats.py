"""Input-only statistics of the RR test sets (no GT, no model): PAN vs MS radiometry and PAN sharpness per sensor."""
import sys, json
from pathlib import Path
import h5py, numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/performance/54_multi_sensor_scale_mixed_training'))
import torch
from prepare import Wald, SENSORS, gen_mtf
torch.set_grad_enabled(False)

def fft_blur(x, k):
    """replicate-pad cross-correlation via FFT (x: C,H,W float64; k: 41x41)."""
    pad = k.shape[-1] // 2; xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='edge'); Hp, Wp = xp.shape[-2:]
    full = np.fft.irfft2(np.fft.rfft2(xp, s=(Hp, Wp)) * np.fft.rfft2(k[::-1, ::-1], s=(Hp, Wp)), s=(Hp, Wp))
    return full[..., 2 * pad:2 * pad + x.shape[-2], 2 * pad:2 * pad + x.shape[-1]]

def high(x, cut=1 / 8):
    h, w = x.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= cut
    return np.real(np.fft.ifft2(np.fft.fft2(x) * f))

def band(x, lo, hi):
    h, w = x.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :])
    return np.real(np.fft.ifft2(np.fft.fft2(x) * ((f >= lo) & (f < hi))))

out = {}
for s in ('QB', 'GF2', 'WV3', 'WV2'):
    tag = SENSORS[s]['tag']; rows = []
    kpan = gen_mtf(SENSORS[s]['pan_gnyq'], 4)
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        keys = list(f.keys()); n = len(f['pan'])
        for i in range(n):
            pan = np.asarray(f['pan'][i], dtype=np.float64); lms = np.asarray(f['lms'][i], dtype=np.float64); ms = np.asarray(f['ms'][i], dtype=np.float64)
            I = ms.mean(0)                                   # MS intensity on the MS grid
            pl = fft_blur(pan, kpan)[0][2::4, 2::4]          # PAN brought to the MS grid (sensor PAN MTF, phase 2)
            pl_box = pan[0].reshape(pan.shape[1] // 4, 4, pan.shape[2] // 4, 4).mean((1, 3))
            A = np.stack([I.ravel(), np.ones(I.size)], 1); (a, b), *_ = np.linalg.lstsq(A, pl.ravel(), rcond=None)   # PAN_L ~ a*I + b
            # multi-band regression PAN_L ~ sum w_b MS_b + c  (how well a linear band mix explains PAN)
            B = np.concatenate([ms.reshape(ms.shape[0], -1).T, np.ones((I.size, 1))], 1); wb, *_ = np.linalg.lstsq(B, pl.ravel(), rcond=None)
            r2 = 1 - ((B @ wb - pl.ravel()) ** 2).sum() / ((pl.ravel() - pl.mean()) ** 2).sum()
            Ilms = lms.mean(0)
            # PAN sharpness relative to MS: energy of PAN in the 4-8 px band (just above the MS Nyquist) and 2-4 px band, relative to PAN energy in 8-16 px
            e816 = (band(pan[0], 1 / 16, 1 / 8) ** 2).mean(); e48 = (band(pan[0], 1 / 8, 1 / 4) ** 2).mean(); e24 = (band(pan[0], 1 / 4, 0.51) ** 2).mean()
            rows.append(dict(ratio_mean=pan.mean() / max(lms.mean(), 1e-6), slope=a, offset=b, std_ratio=pl.std() / max(I.std(), 1e-9), corr=np.corrcoef(pl.ravel(), I.ravel())[0, 1], r2_bands=r2,
                             pan_mean=pan.mean(), ms_mean=ms.mean(), pan_std=pan.std(), ms_int_std=I.std(), pan_hf_48_over_816=np.sqrt(e48 / e816), pan_hf_24_over_816=np.sqrt(e24 / e816),
                             pan_high_std_over_Istd=high(pan[0]).std() / max(Ilms.std(), 1e-9), band_means=[float(v) for v in ms.mean((1, 2))], wb=[float(v) for v in wb]))
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])), float(np.min([r[k] for r in rows])), float(np.max([r[k] for r in rows]))) for k in rows[0] if k not in ('band_means', 'wb')}
    agg['band_means'] = [float(v) for v in np.mean([r['band_means'] for r in rows], 0)]; agg['wb_mean'] = [float(v) for v in np.mean([r['wb'] for r in rows], 0)]
    out[s] = dict(n=n, keys=keys, agg=agg, rows=rows)
    print(s, n, keys)
    for k, v in agg.items():
        if isinstance(v, tuple): print(f'   {k:26s} mean {v[0]:9.4f}  sd {v[1]:8.4f}  min {v[2]:9.4f}  max {v[3]:9.4f}')
        else: print(f'   {k:26s}', np.round(v, 3))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
