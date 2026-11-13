"""Windows controller (evaluation only, no training): `prepare` hashes the test data and the sources and seals the config; `run` evaluates
QB, GF2 and WV3 for seeds 42-44 one at a time on the GPU (WV3 also on WV2) and writes the per-target 3-seed summary. A rerun skips finished
evaluations (evaluation_complete.json)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import hashlib,json,platform,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from evaluate import sha,write,now,run_name,evaluate,summarize,SENSORS,data_files


def source_hashes(conf):
    perf=HERE.parent;e43=perf/'43_normalized_parallel_capacity';e54=perf/'54_multi_sensor_scale_mixed_training';e79=perf/'79_deep_narrow_branch_projection'
    paths=[HERE/'config.json',HERE/'prepare.py',HERE/'evaluate.py',HERE/'run.py',e79/'prepare.py',e54/'model.py',e54/'prepare.py',e54/'evaluate.py',e54/'config.json',
           e43/'model.py',*sorted((e43/'provenance/source').rglob('*.py')),ROOT/'common/interpolation.py',ROOT/'common/metrics.py',ROOT/'common/observation.py',
           ROOT/'common/evaluation_protocol.py',ROOT/'common/result_images.py']
    paths+=[ROOT/r['path'] for r in conf['reproduction_references'].values()]
    paths+=[ROOT/conf[k][s].replace('<seed>',str(x)) for k in ('controls','candidates','original') for s in conf['sensors'] for x in conf['seeds']]
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in dict.fromkeys(paths)}


def fingerprint(cfg):return hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()


def prepare():
    import torch
    from prepare import Projected,ProjectedVariant
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf8'));root=Path(cfg['runtime']);root.mkdir(parents=True,exist_ok=True)
    if (root/'runtime_config.json').exists():raise RuntimeError('runtime already prepared; do not overwrite')
    data={}
    for target,want in cfg['data_sha256'].items():
        for split in ('RR','FR'):
            p=Path(cfg['data_root'])/data_files(target)[split];h=sha(p);print('HASH',target,split,flush=True)
            if h!=want[split]:raise ValueError('test data sha mismatch '+target+' '+split)
            data[f'{target}_{split}']=dict(path=str(p),sha256=h,bytes=p.stat().st_size)
    # The float64 variant with the 79 tolerance and cap must reproduce 79 Projected exactly (same code path).
    torch.manual_seed(0)
    class Net(torch.nn.Module):
        def forward(self,pan,lms,ms):return lms
    checks={}
    for target in cfg['data_sha256']:
        nb=SENSORS[target]['bands'];x=[torch.rand(1,1,128,128,device='cuda'),torch.rand(1,nb,128,128,device='cuda'),torch.rand(1,nb,32,32,device='cuda')]
        a=Projected(Net(),target)(*x);b=ProjectedVariant(Net(),target)(*x);checks[target]=float((a-b).abs().max())
        if checks[target]!=0.0:raise AssertionError('float64 variant differs from 79 Projected '+target)
    cfg['source_hashes']=source_hashes(cfg);cfg['created_kst']=now();cfg['projection_identity_check']=checks;cfg['fingerprint']=fingerprint(cfg)
    write(root/'data_manifest.json',data);write(root/'runtime_config.json',cfg)
    write(root/'environment.json',dict(time_kst=now(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0)))
    print('PREPARE PASS',json.dumps(checks),flush=True)


def run():
    root=Path(json.loads((HERE/'config.json').read_text(encoding='utf8'))['runtime']);cfg=json.loads((root/'runtime_config.json').read_text(encoding='utf8'))
    if fingerprint(cfg)!=cfg['fingerprint'] or source_hashes(cfg)!=cfg['source_hashes']:raise RuntimeError('sealed config/source changed')
    import torch
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(root/'controller_started.json',dict(time_kst=now(),pid=os.getpid(),fingerprint=cfg['fingerprint']))
    queue=[(s,x) for x in cfg['seeds'] for s in cfg['sensors']]
    for sensor,seed in queue:
        print(json.dumps(dict(time_kst=now(),phase='evaluate',sensor=sensor,seed=seed)),flush=True)
        try:evaluate(cfg,sensor,seed)
        except BaseException:
            write(root/'runs'/run_name(sensor,seed)/'failure_evaluate.json',dict(time_kst=now(),traceback=traceback.format_exc()));raise
    out=summarize(cfg)
    write(root/'all_complete.json',dict(time_kst=now(),fingerprint=cfg['fingerprint'],runs=len(queue),reproduction=out['reproduction']))
    print('RUN COMPLETE',json.dumps(out['reproduction']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
    prepare() if a.mode=='prepare' else run()
