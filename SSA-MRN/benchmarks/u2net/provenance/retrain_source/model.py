"""Official U2Net, sole model-source change: constructor MS-band literal 8 -> bands.
The upstream files remain untouched. Dynamic compilation and its exact one-line diff are recorded.
"""
import ast, importlib.util, sys, types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent

def network_class(bands):
    if bands not in (4,8):raise ValueError(bands)
    tree=ast.parse((HERE/'upstream/u2net.py').read_text(encoding='utf-8-sig'))
    targets=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ms_dim' and isinstance(n.value,ast.Constant) and n.value.value==8]
    if len(targets)!=1:raise ValueError('Expected exactly one upstream ms_dim = 8 assignment')
    targets[0].value=ast.Constant(bands);ast.fix_missing_locations(tree)
    old={k:v for k,v in sys.modules.items() if k=='model' or k.startswith('model.')}
    try:
        ns=types.ModuleType('model');ns.__path__=[str(HERE/'upstream')];sys.modules['model']=ns
        scope={'__name__':'u2net_frozen_bands','__file__':str(HERE/'upstream/u2net.py')}
        exec(compile(tree,str(HERE/'upstream/u2net.py'),'exec'),scope)
        return scope['U2Net']
    finally:
        for k in list(sys.modules):
            if k=='model' or k.startswith('model.'):sys.modules.pop(k)
        sys.modules.update(old)

def build_network(bands=4,seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return network_class(bands)(dim=32)

class Adapter(nn.Module):
    def __init__(self,bands=4,seed=42):
        super().__init__();self.net=build_network(bands,seed)
    def forward(self,pan,lms,ms):
        b,c,h,w=ms.shape;H,W=pan.shape[-2:]
        if b!=1 or (H,W)!=(4*h,4*w):raise ValueError('Inference requires batch1 and spatial scale4')
        eh,ew=64-H%64,64-W%64
        mp=ms.new_zeros((1,c,h+2+eh//4,w+2+ew//4));pp=pan.new_zeros((1,1,H+8+eh,W+8+ew))
        mp[:,:,:h+2,:w+2]=F.pad(ms,(1,1,1,1),mode='reflect');pp[:,:,:H+8,:W+8]=F.pad(pan,(4,4,4,4),mode='reflect')
        y=ms.new_zeros((1,c,H+eh,W+ew))
        for i in range((H+eh)//64):
            for j in range((W+ew)//64):
                v=self.net(mp[:,:,i*16:(i+1)*16+2,j*16:(j+1)*16+2],pp[:,:,i*64:(i+1)*64+8,j*64:(j+1)*64+8])
                y[:,:,i*64:(i+1)*64,j*64:(j+1)*64]=v.clamp(0,1)[:,:,4:68,4:68]
        return y[:,:,:H,:W]

def build(bands,sensor,setting,seed):
    if setting=='retrain' and bands==4 and sensor in ('QB','GF2'):return Adapter(4,seed)
    raise ValueError('This package is fresh QB/GF2 retraining, not author inference')
def load(net,path):
    ck=torch.load(path,map_location='cpu',weights_only=False);net.net.load_state_dict(ck['model'],strict=True)
