"""Inference-only W2 output scaling: backbone = original(x) + alpha * branch(x) for 53 C2MS and 43 C2 (QB, seeds 42-44).
Validation uses the 54 trainer formula (FP32, peak 1); RR test uses common/evaluation_protocol.predict and common/metrics.rr_metrics.
Import order follows 77 evaluate.py (model modules first, then common)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import h5py, numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE / 'SSA-MRN'; sys.path.insert(0, str(ROOT))
E = ROOT / 'experiments/performance'
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
M54 = load('m54', E / '54_multi_sensor_scale_mixed_training/model.py'); M43 = M54.M43
from common.metrics import rr_metrics
from common import evaluation_protocol as protocol
p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--alphas', type=float, nargs='+', required=True); p.add_argument('--out', required=True); a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True); torch.set_num_threads(1)
D = Path(a.data); val_path = D / 'QuickBird/Training Dataset/valid_qb.h5'; rr_path = D / 'QuickBird/Testing Dataset (ReducedData, H5 Format)/test_qb_multiExm1.h5'
data_sha = {'val': sha(val_path), 'RR': sha(rr_path)}
assert data_sha == {'val': 'b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470', 'RR': '9842a1232ad2d5b9a61c9fd7fb353e8fc6214eeac6e947d888f221ad6ecdda35'}, data_sha
with h5py.File(rr_path) as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
with h5py.File(val_path) as f: va = {k: torch.from_numpy(np.asarray(f[k], dtype=np.float32) / 2047.) for k in ('pan', 'lms', 'ms', 'gt')}
@torch.inference_mode()
def validation(net):
    se = 0.; el = 0; ps = 0.; n = 0; ang = 0.; pix = 0
    for i in range(0, len(va['pan']), 32):
        b = [va[k][i:i + 32].cuda() for k in ('pan', 'lms', 'ms')]; gt = va['gt'][i:i + 32].cuda(); pr = net(*b).float()
        err = (pr - gt).square(); se += err.double().sum().item(); el += gt.numel(); n += len(gt)
        ps += (-10 * torch.log10(err.mean((2, 3)).clamp_min(1e-15))).mean(1).double().sum().item()
        nm = pr.norm(dim=1) * gt.norm(dim=1); mask = nm > 1e-12; cos = (pr * gt).sum(1) / nm.clamp_min(1e-12)
        ang += torch.acos(cos.clamp(-1, 1))[mask].double().sum().item(); pix += int(mask.sum())
    return dict(val_mse=se / el, val_psnr=ps / n, val_sam=math.degrees(ang / pix))
out = dict(alphas=a.alphas, data_sha256=data_sha, environment=dict(torch=torch.__version__, device=torch.cuda.get_device_name(0)), sweep_sha256=sha(__file__), runs={})
for kind in ('C2MS', 'C2'):
    for seed in (42, 43, 44):
        path = E / ('53_scale_mixed_training/checkpoints/C2MS_QB_s%d/best.pt' % seed if kind == 'C2MS' else '43_normalized_parallel_capacity/checkpoints/C2_QB_s%d/best.pt' % seed)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = M54.Model('C2MS', 4, seed) if kind == 'C2MS' else M43.Model('C2', seed)
        net.load_state_dict(ck['model'], strict=True); net = net.cuda().eval()
        bb = net.core.backbone; state = {'alpha': 1.0}
        bb.forward = lambda x, bb=bb, state=state: bb.original(x) + state['alpha'] * bb.branch(x)
        run = dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path), stored_metrics=ck.get('metrics'), best_epoch=ck.get('best_epoch'), rows={})
        for al in a.alphas:
            state['alpha'] = al; t = time.time(); v = validation(net)
            sc = [rr_metrics(s['gt'], protocol.predict(net, s['pan'], s['lms'], s['ms'], 'cuda')) for s in rr]
            m = {k: float(np.mean([x[k] for x in sc])) for k in sc[0]}
            run['rows'][repr(al)] = dict(val=v, RR=m, RR_scenes={k: [x[k] for x in sc] for k in ('PSNR', 'SAM', 'ERGAS')}, seconds=time.time() - t)
            print(json.dumps(dict(model=kind, seed=seed, alpha=al, val_psnr=round(v['val_psnr'], 4), val_mse=v['val_mse'], RR_PSNR=round(m['PSNR'], 4), RR_SAM=round(m['SAM'], 4), RR_ERGAS=round(m['ERGAS'], 4), sec=round(time.time() - t, 1))), flush=True)
        out['runs'][f'{kind}_s{seed}'] = run; del net; torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
print('ALL_DONE', flush=True)
