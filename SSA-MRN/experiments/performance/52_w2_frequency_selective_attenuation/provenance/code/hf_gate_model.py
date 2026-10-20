"""C2 W2 control versus feature-frequency selective attenuation. No test metadata."""
import torch
from torch import nn
from torch.nn import functional as F
from reference43 import Model as Reference43

EPS=1e-8

def lowpass(x):
    # Feature-domain binomial blur, NOT the sensor MS observation operator.
    with torch.autocast(x.device.type,enabled=False):
        x=x.float();c=x.shape[1];k=x.new_tensor([1,4,6,4,1])/16
        x=F.conv2d(F.pad(x,(2,2,0,0),mode='reflect'),k.view(1,1,1,5).expand(c,1,1,5),groups=c)
        return F.conv2d(F.pad(x,(0,0,2,2),mode='reflect'),k.view(1,1,5,1).expand(c,1,5,1),groups=c)

def energy_ratio(low,high):
    a=low.square().mean((1,2,3));b=high.square().mean((1,2,3))
    return b/(a+b+EPS)

class FrequencyParallel(nn.Module):
    def __init__(self,parallel,arm):
        super().__init__();self.original=parallel.original;self.branch=parallel.branch;self.arm=arm
        # Separate RNG: adding gate parameters cannot perturb the C2 initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261020);self.gate=nn.Sequential(nn.Linear(2,8),nn.SiLU(),nn.Linear(8,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self.last_gate=None;self.last_features=None
    def forward(self,x):
        original=self.original(x);b=self.branch(x)
        if self.arm=='baseline':return original+b
        with torch.autocast(x.device.type,enabled=False):
            low=lowpass(b);high=b.float()-low;xl=lowpass(x)
            # Detached, per-sample scalar features; no batch pooling or GT dependence.
            features=torch.stack((energy_ratio(low,high),energy_ratio(xl,x.float()-xl)),dim=1).detach()
            g=torch.sigmoid(self.gate(features)).view(-1,1,1,1)
            self.last_gate=g.detach();self.last_features=features.detach()
            delta=low+g*high
            return original+delta.to(b.dtype)

class Model(nn.Module):
    def __init__(self,arm='baseline',seed=42):
        super().__init__()
        if arm not in ('baseline','hf_gate'):raise ValueError(arm)
        self.arm=arm;ref=Reference43('C2',seed=seed);self.core=ref.core
        self.core.backbone=FrequencyParallel(self.core.backbone,arm)
    def forward(self,pan,lms,ms):return self.core(pan,lms,ms)
    def gate_snapshot(self):
        b=self.core.backbone
        if b.last_gate is None:return {'applied':False,'scope':'baseline bypass or not yet forwarded'}
        return {'applied':True,'scope':'most recent forward only','mean':float(b.last_gate.mean()),
                'min':float(b.last_gate.min()),'max':float(b.last_gate.max())}

def common_state(model):return model.state_dict()
def load_common(model,state):model.load_state_dict(state,strict=True)
def architecture_spec(arm):
    if arm not in ('baseline','hf_gate'):raise ValueError(arm)
    return {'arm':arm,'base':'43 C2: original N0 plus W2','change':'B_low + sigmoid(MLP(detached relative energies))*B_high' if arm=='hf_gate' else 'original+B; gate bypass',
            'filter':[1,4,6,4,1],'filter_denominator':16,'filter_boundary':'reflect, separable, no downsampling',
            'epsilon':EPS,'gate':'2->8->1 SiLU/sigmoid, per sample, common across all feature channels/pixels',
            'initial_gate':.5,'gate_parameters':33,'initial_outputs_expected_equal':True,
            'baseline_extra_parameters':'same 33 gate parameters present but unused; no gradient/Adam state for them',
            'feature_blur_is_sensor_MTF':False,'parameters':2853951}
