"""Official U2Net: author WV3 weights, original 64px tiled inference."""
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

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = U2Net(dim=32)

    def forward(self, pan, lms, ms):
        # Match upstream test.py including the extra tile at exact multiples.
        b, c, h, w = ms.shape
        H, W = pan.shape[-2:]
        if b != 1 or c != 8 or (H, W) != (h*4, w*4):
            raise ValueError('author U2Net requires batch1, 8 bands, spatial scale4')
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
    if bands != 8 or sensor not in ('WV3','WV2') or setting != 'author':
        raise ValueError('Only official WV3 author weights / WV2 transfer are ready; QB/GF2 are not trained.')
    return Adapter()

def load(net, path):
    net.net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
