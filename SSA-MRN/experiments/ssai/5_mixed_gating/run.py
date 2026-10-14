"""Prepare a new runtime, referencing verified completed experiment4 assets."""
import argparse,json,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from common import ssai_suite as suite
import torch,numpy as np

def prepare(runtime,previous):
    if runtime.exists():raise FileExistsError("Fresh runtime required")
    old=json.loads((previous/'plan.json').read_text())
    current={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0)}
    for k in ('python','executable','torch','numpy','cuda','cudnn','gpu'):
        if current[k]!=old['environment'][k]:raise ValueError('Environment changed: '+k)
    refs={}
    cache=Path(old.get('cache_root',str(previous/'cache')))
    for split in ('train','val'):
        d=old['data'][split]
        if suite.sha(d['path'])!=d['sha256']:raise ValueError('Original data changed')
        for key,h in d['cache'].items():
            if suite.sha(cache/split/(key+'.npy'))!=h:raise ValueError('Cache changed')
    for name in ('common/model.py','references/upstream/network.py'):
        if suite.sha(ROOT/name)!=old['source_hashes'][name]:raise ValueError('Baseline code changed: '+name)
    for arm in ('A0','A2'):
        done=json.loads((previous/arm/'complete.json').read_text())
        refs[arm]={}
        for name in ('best','latest'):
            p=previous/arm/(name+'.pt');h=suite.sha(p)
            if h!=done[name+'_sha256']:raise ValueError('Reference checkpoint changed')
            ck=torch.load(p,map_location='cpu',weights_only=False)
            for k in ('seed','sensor','k','epochs','effective_batch','micro_batch','lr','precision','amp','deterministic','tf32','data','cache_mode'):
                if ck['config']['plan'][k]!=old[k]:raise ValueError('Reference conditions changed: '+k)
            suite.model(arm).load_state_dict(ck['model'],strict=True)
            refs[arm][name]={'path':str(p),'sha256':h,'epoch':ck['epoch']}
    runtime.mkdir(parents=True);p=dict(old)
    p.pop('initial_checkpoints',None);p.pop('optimization_benchmark',None);p.pop('vram_safety',None)
    files=list((ROOT/'common').glob('*.py'))+[ROOT/'references/upstream/network.py']+list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'config.json']
    p.update(arms=['M25','M50'],cache_root=str(cache),reference_runs=refs,environment=current,created_utc=time.time(),source_hashes={x.relative_to(ROOT).as_posix():suite.sha(x)for x in files})
    for x in files:
        dst=runtime/'provenance/source'/x.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(x,dst)
    suite.write(runtime/'plan.json',p);suite.write(runtime/'provenance/plan.json',p)
    print(json.dumps({'prepared':True,'reused_reference_weights':4,'cache_root':str(cache),'environment':current}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--runtime',type=Path,required=True);a.add_argument('--previous',type=Path,required=True);v=a.parse_args();prepare(v.runtime,v.previous)
