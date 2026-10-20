"""Ground-truth based evaluation protocol (criteria v2, experiment 47).

RR       : PanCollection RR test scenes as provided (GT available).
RR2      : RR test scenes degraded once more by the dataset's own Wald operator; GT = the RR MS.
FR_WALD  : FR test scenes degraded by the same operator; GT = the native FR MS.
FR_OBS   : interior RMSE (DN) between the observation of the FR output and the FR MS.
FR QNR / D_lambda / D_s are still computed with common/metrics.fr_metrics but are reported only
(experiment 46: on QB they rank the ground truth below model outputs).

Degradation: MS by common/observation.SensorObservation (DLPan MTF, x4, phase 2; experiment 3 audit),
PAN by the DLPan MTF_PAN filter (QB GNyq 0.15, 41 taps) and the same x4 phase-2 sampling,
LMS by common/interpolation.interp23tap (reproduces the H5 LMS of the test scenes exactly).
"""
import numpy as np
import torch
from torch.nn import functional as F

from common.interpolation import interp23tap
from common.metrics import fr_metrics, rr_metrics
from common.observation import SensorObservation

RATIO = 4
PAN_GNYQ = {'QB': 0.15}
OBS_MARGIN = 5


def _pan_kernel(gain, ratio=RATIO, size=41):
    coordinate = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(coordinate, coordinate, indexing='ij')
    window = np.interp(np.sqrt(x * x + y * y), np.arange(size // 2 + 1), np.kaiser(size, 0.5)[size // 2:], right=0.0)
    alpha = np.sqrt(((size - 1) / (2 * ratio)) ** 2 / (-2 * np.log(gain)))
    desired = np.exp(-(x * x + y * y) / (2 * alpha * alpha))
    desired[desired < np.finfo(desired.dtype).eps * desired.max()] = 0
    desired /= desired.max()
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(desired))) * window)


class WaldDegradation:
    """Dataset-consistent x4 degradation of (PAN, MS) on full scenes."""

    def __init__(self, sensor='QB', device='cpu'):
        self.ms = SensorObservation(sensor, 'decimate', (2, 2)).to(device).double()
        self.pan_kernel = torch.from_numpy(_pan_kernel(PAN_GNYQ[sensor])[None, None]).to(device).double()
        self.device = device

    def pan(self, pan):
        pad = self.pan_kernel.shape[-1] // 2
        rows = torch.arange(-pad, pan.shape[-2] + pad, device=pan.device).clamp(0, pan.shape[-2] - 1)
        cols = torch.arange(-pad, pan.shape[-1] + pad, device=pan.device).clamp(0, pan.shape[-1] - 1)
        blurred = F.conv2d(pan.index_select(-2, rows).index_select(-1, cols), self.pan_kernel)
        return blurred[..., 2::RATIO, 2::RATIO]

    def __call__(self, pan_dn, ms_dn):
        """numpy DN arrays (1,H,W) and (C,h,w) -> degraded (pan, lms, ms) numpy DN arrays."""
        with torch.no_grad():
            p = torch.from_numpy(np.asarray(pan_dn, dtype=np.float64))[None].to(self.device)
            m = torch.from_numpy(np.asarray(ms_dn, dtype=np.float64))[None].to(self.device)
            pan_b = self.pan(p); ms_b = self.ms(m)
            lms_b = interp23tap((ms_b / 2047).float()).double() * 2047
        return pan_b[0].cpu().numpy(), lms_b[0].cpu().numpy(), ms_b[0].cpu().numpy()


def observation_rmse(fused_dn, ms_dn, observation, margin=OBS_MARGIN):
    with torch.no_grad():
        f = torch.from_numpy(np.asarray(fused_dn, dtype=np.float64))[None].to(observation.kernel.device)
        e = observation(f.to(observation.kernel.dtype))[0].cpu().numpy() - np.asarray(ms_dn, dtype=np.float64)
    return float(np.sqrt(np.mean(e[..., margin:-margin, margin:-margin] ** 2)))


def predict(net, pan_dn, lms_dn, ms_dn, device):
    x = [torch.from_numpy((np.asarray(a, dtype=np.float64) / 2047).astype('float32')).unsqueeze(0).to(device) for a in (pan_dn, lms_dn, ms_dn)]
    with torch.no_grad():
        y = net(*x)
    out = y[0].float().cpu().numpy().astype(np.float64) * 2047
    if not np.isfinite(out).all():
        raise FloatingPointError('nonfinite prediction')
    return out


def scene_metrics(net, rr_scene, fr_scene, device, degrade, observation):
    """All criteria-v2 metrics for one RR scene and one FR scene (DN numpy dicts with pan/lms/ms[/gt])."""
    out = {}
    pred = predict(net, rr_scene['pan'], rr_scene['lms'], rr_scene['ms'], device)
    out['RR'] = rr_metrics(rr_scene['gt'], pred)
    pan_b, lms_b, ms_b = degrade(rr_scene['pan'], rr_scene['ms'])
    out['RR2'] = rr_metrics(rr_scene['ms'], predict(net, pan_b, lms_b, ms_b, device))
    pan_b, lms_b, ms_b = degrade(fr_scene['pan'], fr_scene['ms'])
    out['FR_WALD'] = rr_metrics(fr_scene['ms'], predict(net, pan_b, lms_b, ms_b, device))
    fused = predict(net, fr_scene['pan'], fr_scene['lms'], fr_scene['ms'], device)
    out['FR'] = fr_metrics(fused, fr_scene['lms'], fr_scene['ms'], fr_scene['pan'])
    out['FR']['OBS_RMSE_DN'] = observation_rmse(fused, fr_scene['ms'], observation)
    return out
