
import torch
from torch import nn
from torch.nn import functional as F
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.observation import SensorObservation

class FFTObservation(SensorObservation):
    def __init__(self, sensor):
        super().__init__(sensor)
        self.fft_enabled = False
        self._geometry = {}

    def blur(self, prediction):
        if not self.fft_enabled:
            return super().blur(prediction)
        if prediction.ndim != 4 or prediction.shape[1] != self.kernel.shape[0]:
            raise ValueError("Expected sensor-band NCHW")
        h, w = prediction.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError("Observation dimensions must be divisible by four")
        key = (h, w, prediction.device, prediction.dtype)
        if key not in self._geometry:
            from scipy.fft import next_fast_len
            kh, kw = self.kernel.shape[-2:]
            ph, pw = kh // 2, kw // 2
            rows = torch.arange(-ph, h + ph, device=prediction.device).clamp(0, h - 1)
            cols = torch.arange(-pw, w + pw, device=prediction.device).clamp(0, w - 1)
            shape = (next_fast_len(h + 2*ph + kh - 1), next_fast_len(w + 2*pw + kw - 1))
            # conv2d is correlation: reverse the FIR for FFT linear convolution.
            kernel = self.kernel[:, 0].to(dtype=prediction.dtype).flip((-2, -1))
            frequency = torch.fft.rfft2(kernel, s=shape)
            self._geometry[key] = (rows, cols, shape, frequency, kh, kw)
        rows, cols, shape, frequency, kh, kw = self._geometry[key]
        padded = prediction.index_select(-2, rows).index_select(-1, cols)
        full = torch.fft.irfft2(torch.fft.rfft2(padded, s=shape) * frequency, s=shape)
        return full[..., kh-1:kh-1+h, kw-1:kw-1+w]


class SharedThreeStage(nn.Module):
    def __init__(self, channels, k, feedback):
        super().__init__()
        # Instantiate the common core first for identical B1/B2 initialization.
        self.core = RestoredPansharpeningNet(channels, k)
        self.error_encoder = nn.Conv2d(channels, channels, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.observation = FFTObservation("QB").float()
        self.feedback = feedback
        self.margin = 5
        self._masks = {}
        self.register_buffer("phase_candidates", torch.tensor([[2,2],[1,2],[2,1]], dtype=torch.long))

    def samples(self, blurred):
        # Static phase slices keep all selection and indexing on the GPU.
        return torch.stack([blurred[...,2::4,2::4], blurred[...,1::4,2::4],
                            blurred[...,2::4,1::4]], dim=1)

    @torch.no_grad()
    def select_phase(self, lms, ms):
        # Inference geometry uses observed LMS/MS only; never test GT.
        with torch.autocast("cuda", enabled=False):
            views = self.samples(self.observation.blur(lms.float()))
        margin = self.margin if min(ms.shape[-2:]) > 2*self.margin else 0
        delta = views - ms.float()[:,None]
        if margin: delta = delta[...,margin:-margin,margin:-margin]
        return delta.square().mean((2,3,4)).argmin(1)

    def forward(self, pan, lms, ms, phase=None):
        current = lms
        if self.feedback and phase is None: phase = self.select_phase(lms, ms)
        for _ in range(3):
            if self.feedback:
                # Fixed FP32 sensor model, with gradients through current reconstruction.
                with torch.autocast("cuda", enabled=False):
                    views = self.samples(self.observation.blur(current.float()))
                    observed = views[torch.arange(len(ms),device=ms.device),phase]
                    error = ms.float() - observed
                    m = self.margin
                    geometry = (ms.shape[-2], ms.shape[-1], ms.device)
                    if geometry not in self._masks:
                        rows = torch.arange(ms.shape[-2],device=ms.device)
                        cols = torch.arange(ms.shape[-1],device=ms.device)
                        self._masks[geometry] = ((rows>=m)&(rows<ms.shape[-2]-m))[:,None] & ((cols>=m)&(cols<ms.shape[-1]-m))[None,:]
                    mask = self._masks[geometry]
                    error = F.interpolate(error*mask, size=current.shape[-2:],mode="bilinear",align_corners=False)
            else:
                error = torch.zeros_like(current)
            conditioned = current + self.error_encoder(error)
            proposal = self.core(pan, conditioned, ms)
            current = current + self.alpha * (proposal - current)
        return current

def build_model(variant, channels, k):
    if variant not in ("B1","B2"): raise ValueError(variant)
    return SharedThreeStage(channels,k,variant=="B2")

def train_config(variant, seed):
    return {"variant":variant,"seed":seed,"stages":3,"shared_weights":True,
            "loss":"final_output_MSE","initial_alpha":0.1,
            "train_phase":"audited original TRAIN GT/MS",
            "inference_phase":"observed LMS/MS, never GT"}
