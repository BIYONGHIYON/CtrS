"""Load any retained experiment checkpoint and evaluate an explicit H5 test file.

Only load trusted repository checkpoints: full training state uses pickle.
This tool performs FP32 inference; it never starts or resumes training.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def experiment(number):
    matches = list((ROOT / 'experiments').glob(f'{number}_*'))
    if len(matches) != 1:
        raise ValueError(f'Unknown experiment number: {number}')
    return matches[0]


def model_module(number):
    return importlib.import_module('experiments.' + experiment(number).name + '.model')


def load_model(number, checkpoint):
    import torch
    from common.model import RestoredPansharpeningNet
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {})
    sensor = cfg.get('sensor', ck.get('sensor', 'QB'))
    channels = cfg.get('channels', ck.get('channels', 8 if sensor in ('WV2', 'WV3') else 4))
    k = cfg.get('k', ck.get('ssai_dimension', 4))
    variant = ck.get('variant', cfg.get('variant'))
    if number in (1, 2, 8, 12, 14):
        net = RestoredPansharpeningNet(channels, k)
    elif number in (4, 5, 6):
        net = model_module(4).make_model(cfg)
    elif number in (7, 10, 13, 16):
        net = model_module(number).build_model(variant, channels, k)
    elif number == 9:
        net = model_module(9).FusionPansharpeningNet(channels, k, variant, cfg.get('chunk_rows', 32))
    elif number == 11:
        net = model_module(11).WaveletMS(variant, channels, cfg['width'], cfg['blocks'])
    elif number == 15:
        net = model_module(15).build(ck['arm'])
    elif number in (17, 18, 19, 20, 21, 22):
        cls = {17:'StructureNet',18:'CoarseAnchorNet',19:'DualStreamNet',20:'GuidanceNet',21:'RefinementNet',22:'StageAwareNet'}[number]
        net = getattr(model_module(number), cls)(variant)
    else:
        raise ValueError('This experiment has no trained checkpoint')
    net.load_state_dict(ck['model'], strict=True)
    net.eval()
    return net, ck


def predict(number, model, pan, lms, ms):
    return model(pan, ms) if number == 11 else model(pan, lms, ms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Path relative to experiment folder, or absolute')
    p.add_argument('--h5', type=Path, help='Explicit RR or FR H5, including flattened Kaggle filenames')
    p.add_argument('--protocol', choices=('RR', 'FR'), default='RR')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, help='Fresh metrics JSON path; existing files are never overwritten')
    p.add_argument('--max-samples', type=int, help='Omit to evaluate the entire H5')
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    path = args.checkpoint if args.checkpoint.is_absolute() else experiment(args.experiment) / args.checkpoint
    net, ck = load_model(args.experiment, path)
    net.to(args.device)
    if args.h5 is None:
        cfg = ck.get('config', {})
        sensor = cfg.get('sensor', ck.get('sensor', 'QB'))
        channels = cfg.get('channels', ck.get('channels', 8 if sensor in ('WV2','WV3') else 4))
        with torch.inference_mode():
            y = predict(args.experiment, net, torch.ones(1,1,64,64,device=args.device),
                        torch.ones(1,channels,64,64,device=args.device),torch.ones(1,channels,16,16,device=args.device))
        if not torch.isfinite(y).all():
            raise ValueError('Non-finite output')
        print(json.dumps(dict(strict_load=True,finite_forward=True,shape=list(y.shape),epoch=ck.get('epoch'))))
        return
    if args.output is None:
        p.error('--output is required with --h5')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_samples is not None and args.max_samples < 1:
        p.error('--max-samples must be positive')
    import h5py
    import numpy as np
    from common.data import SENSOR_MAX
    from common.metrics import rr_metrics, fr_metrics
    cfg=ck.get('config',{});sensor=cfg.get('sensor',ck.get('sensor','QB'));scale=SENSOR_MAX[sensor]
    rows=[]
    with h5py.File(args.h5,'r') as f, torch.inference_mode():
        count=min(len(f['pan']),args.max_samples or len(f['pan']))
        for i in range(count):
            keys=('pan','lms','ms','gt') if args.protocol=='RR' else ('pan','lms','ms')
            a={key:np.asarray(f[key][i],dtype=np.float32) for key in keys}
            x=[torch.from_numpy(a[key]/scale).unsqueeze(0).to(args.device) for key in ('pan','lms','ms')]
            pred=predict(args.experiment,net,*x)[0].cpu().numpy().astype(np.float64)*scale
            metrics=rr_metrics(a['gt'].astype(np.float64),pred) if args.protocol=='RR' else fr_metrics(pred,a['lms'],a['ms'],a['pan'])
            if not all(np.isfinite(v) for v in metrics.values()):raise ValueError(f'Non-finite metrics: {i}')
            rows.append(dict(index=i,metrics=metrics));print(f'{i+1}/{count}',flush=True)
    report=dict(experiment=args.experiment,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epoch=ck.get('epoch'),protocol=args.protocol,samples=count,rows=rows,mean={k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']},limits='FP32 portable re-evaluation. QNR/Q2n/SCC MATLAB parity is unverified. Historical evaluations may use additional metrics or a different implementation; this output does not replace them.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
