"""Official author 8-band and retained retrain 4-band U2Net; unchanged tiled inference.

Four-band construction replaces only the official constructor's ms_dim=8 literal
in memory. The upstream submodule and every other network operation stay unchanged.
"""
import ast
import sys, types, importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parent / 'upstream'
saved = {k:v for k,v in sys.modules.items() if k=='model' or k.startswith('model.')}
try:
    pkg=types.ModuleType('model');pkg.__path__=[str(UPSTREAM/'model')]
    sys.modules['model']=pkg
    spec=importlib.util.spec_from_file_location('model.u2net',UPSTREAM/'model/u2net.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    U2Net=module.U2Net
finally:
    for k in list(sys.modules):
        if k=='model' or k.startswith('model.'):sys.modules.pop(k)
    sys.modules.update(saved)

def four_band_class():
    source = UPSTREAM / 'model/u2net.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets)==1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id=='ms_dim'
               and isinstance(n.value, ast.Constant) and n.value.value==8]
    if len(targets)!=1: raise ValueError('Expected exactly one official ms_dim=8 assignment')
    targets[0].value = ast.Constant(4); ast.fix_missing_locations(tree)
    previous = {k:v for k,v in sys.modules.items() if k=='model' or k.startswith('model.')}
    try:
        pkg = types.ModuleType('model'); pkg.__path__=[str(UPSTREAM/'model')]; sys.modules['model']=pkg
        scope={'__name__':'u2net_retained_four_band','__file__':str(source)}
        exec(compile(tree,str(source),'exec'),scope)
        return scope['U2Net']
    finally:
        for name in list(sys.modules):
            if name=='model' or name.startswith('model.'): sys.modules.pop(name)
        sys.modules.update(previous)


class Adapter(nn.Module):
    def __init__(self, bands=8, seed=None):
        super().__init__()
        if bands not in (4,8): raise ValueError('Only 4-band retrain and 8-band author weights are supported')
        self.bands = bands
        with torch.random.fork_rng(devices=[]):
            if seed is not None: torch.manual_seed(seed)
            self.net = (U2Net if bands==8 else four_band_class())(dim=32)

    def forward(self, pan, lms, ms):
        # Match upstream test.py including the extra tile at exact multiples.
        b, c, h, w = ms.shape
        H, W = pan.shape[-2:]
        if b != 1 or c != self.bands or (H, W) != (h*4, w*4):
            raise ValueError('U2Net requires batch1, matching bands, spatial scale4')
        edge_H, edge_W = 64-H%64, 64-W%64
        m = ms.new_zeros((1,c,h+2+edge_H//4,w+2+edge_W//4))
        p = pan.new_zeros((1,1,H+8+edge_H,W+8+edge_W))
        m[:,:,:h+2,:w+2] = F.pad(ms,(1,1,1,1),mode='reflect')
        p[:,:,:H+8,:W+8] = F.pad(pan,(4,4,4,4),mode='reflect')
        out = ms.new_zeros((1,c,H+edge_H,W+edge_W))
        for i in range((H+edge_H)//64):
            for j in range((W+edge_W)//64):
                y = self.net(m[:,:,i*16:(i+1)*16+2,j*16:(j+1)*16+2],
                             p[:,:,i*64:(i+1)*64+8,j*64:(j+1)*64+8])
                out[:,:,i*64:(i+1)*64,j*64:(j+1)*64] = y.clamp(0,1)[:,:,4:68,4:68]
        return out[:,:,:H,:W]

def build(bands, sensor, setting, seed):
    if bands==8 and sensor in ('WV3','WV2') and setting=='author': return Adapter(8)
    if bands==4 and sensor in ('QB','GF2') and setting=='retrain' and seed==42: return Adapter(4,42)
    raise ValueError('Expected author WV3/WV2 (8 bands), or retrain QB/GF2 seed42 (4 bands)')

def load(net, path):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    state = checkpoint['model'] if isinstance(checkpoint,dict) and 'model' in checkpoint else checkpoint
    net.net.load_state_dict(state,strict=True)
