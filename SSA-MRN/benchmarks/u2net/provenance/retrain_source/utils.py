import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');t.replace(p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(x,t);t.replace(p)
def tensors_hash(sd):
    h=hashlib.sha256()
    for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def batch_order(n,seed,epoch,batch=16):
    g=torch.Generator().manual_seed(seed+100000+epoch)
    order=torch.randperm(n,generator=g).numpy();used=order[:n//batch*batch]
    return used,hashlib.sha256(used.tobytes()).hexdigest()
def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r):
    torch.set_rng_state(r['torch']);np.random.set_state(r['numpy']);random.setstate(r['python'])
    if r['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
