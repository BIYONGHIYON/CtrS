import json
from pathlib import Path
import h5py,numpy as np
from utils import digest,write
HERE=Path(__file__).resolve().parent

def find(root,name):
    ps=[p for p in Path(root).rglob('*.h5') if (p.name==name or p.name.endswith('__'+name)) and not p.name.startswith('._') and '__._' not in p.name]
    if len(ps)!=1:raise ValueError(f'{name}: found {len(ps)} files; narrow INPUT path instead of selecting arbitrarily')
    return ps[0]

def prepare(root,cache,sensor):
    cache=Path(cache)/sensor;cache.mkdir(parents=True,exist_ok=True)
    reference=json.loads((HERE/'expected_data_manifest.json').read_text(encoding='utf-8-sig'))['files'];info={}
    scale=1023 if sensor=='GF2' else 2047
    for split,prefix in [('train','train'),('val','valid')]:
        name=f'{prefix}_{sensor.lower()}.h5';p=find(root,name)
        print(f'{sensor} {split}: SHA-256 checking',flush=True);h=digest(p)
        ref=[r for r in reference if r['original_path'].endswith('/'+name)]
        if len(ref)!=1 or h!=ref[0]['sha256']:raise ValueError('Data differs from supplied manifest: '+str(p))
        folder=cache/split;marker=folder/'cache.json'
        if marker.exists():
            c=json.loads(marker.read_text(encoding='utf8'))
            if c['source_sha256']!=h or c['scale']!=scale:raise ValueError('Existing cache differs; use a new cache directory')
            for k in ('gt','pan','ms'):
                a=np.load(folder/(k+'.npy'),mmap_mode='r');assert list(a.shape)==c['shapes'][k]
                if digest(folder/(k+'.npy'))!=c['npy_sha256'][k]:raise ValueError('Cache bytes changed: '+k)
            info[split]=c;continue
        owner=dict(source_sha256=h,scale=scale,generated_cache=True)
        if folder.exists():
            owned=folder/'owner.json'
            if folder.resolve().parent!=cache.resolve() or not owned.exists() or json.loads(owned.read_text(encoding='utf8'))!=owner:
                raise FileExistsError('Unrecognized partial cache; nothing overwritten: '+str(folder))
            print(f'{sensor} {split}: rebuilding this run-owned incomplete cache',flush=True)
        else:folder.mkdir()
        write(folder/'owner.json',owner);shapes={};ranges={}
        with h5py.File(p,'r') as f:
            n=len(f['gt']);assert f['gt'].shape==(n,4,64,64) and f['pan'].shape==(n,1,64,64) and f['ms'].shape==(n,4,16,16)
            for k in ('gt','pan','ms'):
                shapes[k]=list(f[k].shape);a=np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float32',shape=f[k].shape);lo=float('inf');hi=-lo
                for start in range(0,n,128):
                    x=np.asarray(f[k][start:start+128],dtype=np.float32)
                    if not np.isfinite(x).all():raise FloatingPointError(sensor+' '+split+' '+k)
                    lo=min(lo,float(x.min()));hi=max(hi,float(x.max()));a[start:start+len(x)]=x/scale
                a.flush();del a;ranges[k]=[lo,hi]
        c=dict(source=str(p),source_sha256=h,scale=scale,samples=n,shapes=shapes,raw_ranges=ranges,normalization='FP32 DN/scale; no clipping')
        c['npy_sha256']={k:digest(folder/(k+'.npy')) for k in ('gt','pan','ms')}
        write(marker,c);info[split]=c
    return info

def arrays(cache,sensor,split):
    p=Path(cache)/sensor/split
    return {k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('gt','pan','ms')}
