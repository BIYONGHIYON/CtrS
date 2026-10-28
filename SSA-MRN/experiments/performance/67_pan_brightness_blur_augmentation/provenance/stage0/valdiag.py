"""Stage-0 diagnostic on WV3 validation (no test data): sensitivity of WV3-trained C2MS / 58 / 65 to a darker PAN and to a blurrier PAN.
PAN gain 0.70 = GF2/WV3 training ratio (0.82/1.16); extra Gaussian PAN blur sigma 0.88 px lowers the PAN MTF at the MS Nyquist by 0.786."""
import json, sys
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn.functional as F
code = Path(sys.argv[1]); data = Path(sys.argv[2]); run = Path(sys.argv[3]); out = Path(sys.argv[4])
sys.path.insert(0, str(code)); sys.path.insert(0, str(code / 'experiments/performance/65_l1_spectral_loss'))
from model import Model
torch.set_grad_enabled(False); torch.set_num_threads(4)
S = 2047.; N = 512
P = code / 'experiments/performance'
MODELS = {'C2MS': P / '54_multi_sensor_scale_mixed_training/checkpoints/C2MS_WV3_s42/best.pt',
          'SAM58': P / '58_spectral_loss_multi_sensor/checkpoints/C2MS_SAM_WV3_s42/best.pt',
          'L1S65': run / 'runs/WV3_C2MS_L1S_s42/checkpoints/best.pt'}


def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False); m = Model('C2MS', 8, int(ck['seed'])); m.load_state_dict(ck['model'], strict=True); return m.eval()


def gblur(x, s):
    r = int(np.ceil(3 * s)); t = torch.arange(-r, r + 1, dtype=x.dtype); k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode='replicate'); return F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))


def high(x):
    h, w = x.shape[-2:]; f = torch.maximum(torch.fft.fftfreq(h).abs()[:, None], torch.fft.fftfreq(w).abs()[None, :])
    return torch.fft.ifft2(torch.fft.fft2(x) * (f >= 1 / 8)).real


with h5py.File(data / 'WorldView 3/Training Dataset/valid_wv3.h5', 'r') as f:
    idx = np.linspace(0, len(f['pan']) - 1, N).astype(int)
    a = {k: torch.from_numpy(np.asarray(f[k][np.sort(idx)], dtype=np.float32) / S) for k in ('pan', 'lms', 'ms', 'gt')}
conds = {'orig': a['pan'], 'pan_gain_0.70': a['pan'] * 0.70, 'pan_blur_0.88px': gblur(a['pan'], 0.88), 'gain_0.70+blur': gblur(a['pan'], 0.88) * 0.70}
res = {}
for name, path in MODELS.items():
    m = load(path); res[name] = {}
    for c, pan in conds.items():
        ys = torch.cat([m(pan[i:i + 64], a['lms'][i:i + 64], a['ms'][i:i + 64]) for i in range(0, N, 64)]).double(); g = a['gt'].double()
        mse = ((ys - g) ** 2).mean((1, 2, 3)); psnr = float((10 * torch.log10(1 / mse)).mean())
        cos = (ys * g).sum(1) / (ys.norm(dim=1) * g.norm(dim=1) + 1e-12); sam = float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).mean())
        hy, hg = high(ys), high(g); energy = float(((hy ** 2).mean() / (hg ** 2).mean()).sqrt())
        inj = float((ys - a['lms']).std() / (g - a['lms']).std()); bias = float(((ys - g).mean((0, 2, 3))).abs().mean() * S)
        res[name][c] = dict(psnr=psnr, sam=sam, energy_high=energy, injected_std_ratio=inj, band_bias_dn=bias)
        print(name, c, {k: round(v, 4) for k, v in res[name][c].items()}, flush=True)
out.write_text(json.dumps(res, indent=1))
