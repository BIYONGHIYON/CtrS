"""Experiment33 N1 vs local MS-anchored reversible normalization; no learned parameters added."""
import torch
from torch.nn import functional as F
from normalization33 import Model as ArchivedModel


class Model(ArchivedModel):
    def __init__(self, arm='global', std_floor=0.01, window=5):
        if arm not in ('global','local') or window!=5:
            raise ValueError('Prespecified arms are global/local with LR-MS window5')
        super().__init__('N1',std_floor)
        self.mode=arm
        self.window=window
        self.last_statistics=None

    def statistics(self,ms):
        with torch.autocast(device_type=ms.device.type,enabled=False):
            x=ms.float()
            if self.mode=='global':
                # Call the actual archived N1 implementation, without an alternate formula.
                mean,scale=super().statistics(ms)
                raw=torch.var(x,dim=(1,2,3),correction=0,keepdim=True).clamp_min(0).sqrt()
            else:
                if min(x.shape[-2:])<=self.window//2:
                    raise ValueError('LR MS is too small for reflect-padded window5')
                pad=self.window//2
                first=x.mean(1,keepdim=True)
                second=x.square().mean(1,keepdim=True)
                mean=F.avg_pool2d(F.pad(first,(pad,pad,pad,pad),mode='reflect'),self.window,stride=1)
                moment2=F.avg_pool2d(F.pad(second,(pad,pad,pad,pad),mode='reflect'),self.window,stride=1)
                variance=(moment2-mean.square()).clamp_min(0)
                raw=variance.detach().sqrt()
                # Clamp variance before sqrt: finite derivatives even for flat input.
                scale=variance.clamp_min(self.std_floor**2).sqrt()
            if not self.training:
                self.last_statistics=(mean.detach(),raw.detach(),scale.detach())
        return mean,scale

    def fields(self,pan,lms,ms):
        mean,scale=self.statistics(ms)
        if self.mode=='global':return mean,scale,mean,scale
        # Fixed LR pixel-center to HR pixel-center mapping, no input/LMS resampling.
        hr_mean=F.interpolate(mean,size=lms.shape[-2:],mode='bilinear',align_corners=False)
        hr_scale=F.interpolate(scale,size=lms.shape[-2:],mode='bilinear',align_corners=False)
        return mean,scale,hr_mean,hr_scale

    def forward(self,pan,lms,ms):
        if self.mode=='global':
            return super().forward(pan,lms,ms)
        with torch.autocast(device_type=ms.device.type,enabled=False):
            mean,scale,hr_mean,hr_scale=self.fields(pan,lms,ms)
            inputs=((pan.float()-hr_mean)/hr_scale,(lms.float()-hr_mean)/hr_scale,(ms.float()-mean)/scale)
        prediction=self.core(*inputs)
        with torch.autocast(device_type=ms.device.type,enabled=False):
            return prediction.float()*hr_scale+hr_mean


def common_state(model):return model.state_dict()
def load_common(model,state):model.load_state_dict(state,strict=True)
