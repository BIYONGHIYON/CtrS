import torch
from torch import nn
from torch.nn import functional as F
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class FixedHighFrequency(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension):
        super().__init__(channels, ssai_dimension)
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, channels, 3, padding=1))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)

    @staticmethod
    def highpass(pan):
        w = pan.new_tensor([1, 4, 6, 4, 1]) / 16
        kernel = (w[:, None] * w[None, :]).view(1, 1, 5, 5)
        rows = torch.arange(-2, pan.shape[-2] + 2, device=pan.device).clamp(0, pan.shape[-2] - 1)
        cols = torch.arange(-2, pan.shape[-1] + 2, device=pan.device).clamp(0, pan.shape[-1] - 1)
        smooth = F.conv2d(pan.index_select(-2, rows).index_select(-1, cols), kernel)
        return pan - smooth

    def forward(self, pan, lms, ms):
        return super().forward(pan, lms, ms) + self.detail(self.highpass(pan))

class BandGatedHighFrequency(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension):
        super().__init__(channels, ssai_dimension)
        self.band_count = channels
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, channels, 3, padding=1))
            self.gate = nn.Sequential(nn.Conv2d(3 * channels, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, channels, 1))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.0)

    @staticmethod
    def highpass(pan):
        w = pan.new_tensor([1, 4, 6, 4, 1]) / 16
        kernel = (w[:, None] * w[None, :]).view(1, 1, 5, 5)
        rows = torch.arange(-2, pan.shape[-2] + 2, device=pan.device).clamp(0, pan.shape[-2] - 1)
        cols = torch.arange(-2, pan.shape[-1] + 2, device=pan.device).clamp(0, pan.shape[-1] - 1)
        smooth = F.conv2d(pan.index_select(-2, rows).index_select(-1, cols), kernel)
        return pan - smooth

    def forward_components(self, pan, lms, ms):
        base = super().forward(pan, lms, ms)
        high = self.highpass(pan).expand(-1, self.band_count, -1, -1)
        detached = base.detach()
        dx = F.pad(detached[..., 1:] - detached[..., :-1], (0, 1, 0, 0))
        dy = F.pad(detached[..., 1:, :] - detached[..., :-1, :], (0, 0, 0, 1))
        gate = torch.sigmoid(self.gate(torch.cat([high, dx.abs(), dy.abs()], dim=1)))
        correction = gate * self.detail(self.highpass(pan))
        return base, gate, correction

    def forward(self, pan, lms, ms):
        base, _, correction = self.forward_components(pan, lms, ms)
        return base + correction

def build_model(variant, channels, k):
    if variant == "baseline":
        return RestoredPansharpeningNet(channels, k)
    if variant == "high_frequency":
        return FixedHighFrequency(channels, k)
    if variant == "band_gated_hf":
        return BandGatedHighFrequency(channels, k)
    raise ValueError(variant)

def train_config(variant, seed):
    return {"sensor": SENSOR, "variant": variant, "k": K, "seed": int(seed), "epochs": EPOCHS,
            "effective_batch": EFFECTIVE_BATCH_SIZE, "micro_batch": MICRO_BATCH_SIZE,
            "lr": LEARNING_RATE, "loss": "MSE", "train": str(TRAIN_H5), "val": str(VAL_H5),
            "precision": "FP32"}
