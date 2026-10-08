"""Single-change QB PAN ablations; observed inputs only, no GT in correction."""
import torch
from torch import nn
from torch.nn import functional as F
from .ssa_mrn import RestoredPansharpeningNet
from .interp23 import Interp23
from ssamrn.observation import SensorObservation


class Internal23(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension, scope='all'):
        super().__init__(channels, ssai_dimension)
        if scope not in ('all','input','output'):
            raise ValueError('Unknown 23tap scope')
        if scope in ('all','input'):
            self.upsample1 = Interp23(4)
            self.upsample100 = Interp23(2)
        if scope in ('all','output'):
            self.upsample101 = Interp23(2)
            self.upsample102 = Interp23(2)


class LRCorrection(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension, sensor='QB', gain=.1):
        super().__init__(channels, ssai_dimension)
        if sensor != 'QB' or channels != 4:
            raise ValueError('First LR correction experiment is QB only')
        self.observation = SensorObservation(sensor).float()
        self.gain = gain
        self.margin = 5  # Avoid unknown patch halos of the 41x41 MTF kernel.

    def correction(self, prediction, lms, ms):
        margin = self.margin
        if min(ms.shape[-2:]) <= 2*margin:
            raise ValueError('LR patches must retain an interior after halo exclusion')
        # Estimate orientation/sampling from observed LMS and MS, never GT or prediction.
        with torch.no_grad():
            reference = self.observation.blur(lms)
            scores = []
            phases = [(r,c) for r in range(4) for c in range(4)]
            for r,c in phases:
                sample = reference[...,r::4,c::4]
                scores.append((sample[...,margin:-margin,margin:-margin]-ms[...,margin:-margin,margin:-margin]).square().mean((1,2,3)))
            selected = torch.stack(scores,1).argmin(1)
        blurred = self.observation.blur(prediction)
        observations = torch.stack([blurred[...,r::4,c::4] for r,c in phases],1)
        observed = observations[torch.arange(len(prediction),device=prediction.device),selected]
        residual = ms-observed
        rows = torch.arange(ms.shape[-2],device=ms.device)
        cols = torch.arange(ms.shape[-1],device=ms.device)
        mask = ((rows>=margin)&(rows<ms.shape[-2]-margin))[:,None] & ((cols>=margin)&(cols<ms.shape[-1]-margin))[None,:]
        return self.gain*F.interpolate(residual*mask, size=prediction.shape[-2:],mode='bilinear',align_corners=True)

    def forward(self, pan, lms, ms):
        prediction = super().forward(pan,lms,ms)
        return prediction+self.correction(prediction,lms,ms)


class HighFrequency(RestoredPansharpeningNet):
    def __init__(self, channels, ssai_dimension):
        super().__init__(channels, ssai_dimension)
        # Isolate new-layer RNG from the existing core and loader sequence.
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.Conv2d(16,channels,3,padding=1))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)

    @staticmethod
    def highpass(pan):
        weights = pan.new_tensor([1,4,6,4,1])/16
        kernel = (weights[:,None]*weights[None,:]).view(1,1,5,5)
        rows = torch.arange(-2,pan.shape[-2]+2,device=pan.device).clamp(0,pan.shape[-2]-1)
        cols = torch.arange(-2,pan.shape[-1]+2,device=pan.device).clamp(0,pan.shape[-1]-1)
        smooth = F.conv2d(pan.index_select(-2,rows).index_select(-1,cols),kernel)
        return pan-smooth

    def forward(self,pan,lms,ms):
        return super().forward(pan,lms,ms)+self.detail(self.highpass(pan))


def make_model(config):
    channels={'QB':4,'GF2':4,'WV3':8}[config['sensor']]
    arguments = dict(channels=channels,ssai_dimension=config['k'])
    variant = config['variant']
    if variant == 'baseline':
        return RestoredPansharpeningNet(**arguments)
    if variant in ('interp23','interp23_input','interp23_output'):
        scope={'interp23':'all','interp23_input':'input','interp23_output':'output'}[variant]
        return Internal23(**arguments,scope=scope)
    if variant == 'lr_correction':
        return LRCorrection(**arguments,sensor=config['sensor'],gain=config['lr_gain'])
    if variant == 'high_frequency':
        return HighFrequency(**arguments)
    raise ValueError(f'Unknown model variant: {variant}')
