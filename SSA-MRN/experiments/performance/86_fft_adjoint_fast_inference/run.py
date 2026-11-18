"""86 (no training): finish the fast K10 projection and measure inference settings for C2MS-SP, on the server GPU with nothing else running.

  identity  FastProjectorV2 vs the 79/83 projection on the 65 outputs, every RR/FR test scene, seeds 42-44, QB/GF2/WV3 and the WV2 transfer
  profile   torch.profiler of the C2MS network (65, seed 42) at RR 256^2 and FR 512^2, QB and WV3, official setting
  timing    the fixed latency protocol (as 85) for C2MS-SP with the 79, 85 and 86 projections and C2MS-SP-Lite, under each inference setting
  accuracy  RR test change of C2MS-SP (65 + 86 projection, seed 42) under each non-official setting vs the official one, QB/GF2/WV3
`python run.py --mode all` runs the four in this order and writes results/*.json into the runtime folder."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, hashlib, importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, torch, h5py

HERE = Path(__file__).resolve().parent; PERF = HERE.parent; ROOT = PERF.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from prepare import SENSORS, data_files, Projector, Projected, FastProjected, FastProjectorV2, FastProjectedV2, Graphed
from common.metrics import rr_metrics
KST = timezone(timedelta(hours=9))
def now(): return datetime.now(KST).isoformat()
def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, PERF / rel); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
M83 = _mod('exp83_model', '83_factorized_c2ms_finetuning/model.py'); M54 = M83.M54
V = _mod('c2ms54_eval', '54_multi_sensor_scale_mixed_training/evaluate.py')
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8')); OUT = Path(CFG['runtime'])


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / name; t = p.with_suffix('.tmp'); t.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf8'); os.replace(t, p)


def load(key, sensor, seed):
    path = ROOT / CFG[key][sensor].replace('<seed>', str(seed)); ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['seed'] != seed: raise ValueError('seed mismatch ' + str(path))
    m = M54.Model('C2MS', SENSORS[sensor]['bands'], seed) if key == 'c2ms_sp' else M83.Model('K2', SENSORS[sensor]['bands'], seed)
    m.load_state_dict(ck['model'], strict=True); return m.cuda().eval()


def setting(name):
    """official = 81/85 latency setting; the others are inference options (reference)."""
    s = CFG['settings'][name]; det = s['deterministic']
    torch.backends.cudnn.deterministic = det; torch.backends.cudnn.benchmark = s['cudnn_benchmark']
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = s['tf32']; torch.use_deterministic_algorithms(det, warn_only=True)
    return s


def identity():
    rule = CFG['identity']; rows = {}; worst = worst_db = 0.; setting('official')
    for sensor in CFG['sensors']:
        for seed in CFG['seeds']:
            net = load('c2ms_sp', sensor, seed)
            for target in [sensor] + CFG['transfer'].get(sensor, []):
                ref, fast = Projector(target, max_iter=10), FastProjectorV2(target, 'cuda'); diffs, psnr, same = [], [[], []], True
                for split in ('RR', 'FR'):
                    with h5py.File(Path(CFG['data_root']) / data_files(target)[split]) as f:
                        mv = SENSORS[target]['max_value']
                        for i in range(len(f['pan'])):
                            x = [torch.from_numpy((np.asarray(f[k][i], dtype=np.float64) / mv).astype('float32'))[None].cuda() for k in ('pan', 'lms', 'ms')]
                            with torch.inference_mode(): y = net(*x).double()
                            with torch.inference_mode(False): a, ia = ref(y.clone(), x[2].double().clone())
                            with torch.inference_mode(): b, ib = fast(y, x[2].double())
                            diffs.append(float((a.detach() - b).abs().max())); same &= ia == ib
                            if split == 'RR':
                                gt = np.asarray(f['gt'][i], dtype=np.float64)
                                for j, o in enumerate((a.detach(), b)): psnr[j].append(rr_metrics(gt, o[0].float().cpu().numpy().astype(np.float64) * mv)['PSNR'])
                d = float(np.mean(psnr[1]) - np.mean(psnr[0])); worst = max(worst, max(diffs)); worst_db = max(worst_db, abs(d))
                rows[f'{sensor}_s{seed}_{target}'] = dict(max_abs=max(diffs), psnr_diff_db=d, iterations_equal=same)
                print(json.dumps(dict(time_kst=now(), identity=f'{sensor}_s{seed}_{target}', max_abs=max(diffs), psnr_diff_db=d)), flush=True)
    res = dict(time_kst=now(), rule=rule, rows=rows, worst_max_abs=worst, worst_psnr_diff_db=worst_db,
               passed=worst <= rule['tol_max_abs'] and worst_db <= rule['tol_psnr_db'] and all(r['iterations_equal'] for r in rows.values()))
    write('identity.json', res); print('IDENTITY', res['passed'], worst, worst_db, flush=True); return res


def inputs(sensor):
    mv = SENSORS[sensor]['max_value']; out = {}
    for split, size in (('RR', '256'), ('FR', '512')):
        with h5py.File(Path(CFG['data_root']) / data_files(sensor)[split]) as f:
            out[f'{split}_{size}'] = [torch.from_numpy((np.asarray(f[k][0], dtype=np.float64) / mv).astype('float32'))[None].cuda() for k in ('pan', 'lms', 'ms')]
    return out


def profile():
    from torch.profiler import profile as prof, ProfilerActivity
    setting('official'); res = {}
    for sensor in CFG['latency_sensors']:
        net = load('c2ms_sp', sensor, CFG['latency_seed'])
        for size, x in inputs(sensor).items():
            with torch.inference_mode():
                for _ in range(5): net(*x)
                torch.cuda.synchronize()
                with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    for _ in range(10): net(*x)
                    torch.cuda.synchronize()
            ev = [e for e in p.key_averages() if e.device_time_total > 0 or e.cpu_time_total > 0]
            dev = sum(e.self_device_time_total for e in ev) / 10; cpu = sum(e.self_cpu_time_total for e in ev) / 10
            top = sorted(ev, key=lambda e: -e.self_device_time_total)[:15]
            kernels = sum(e.count for e in p.events() if e.device_type.name == 'CUDA') / 10
            res[f'{sensor}_{size}'] = dict(device_ms=dev / 1e3, host_ms=cpu / 1e3, cuda_kernels_per_forward=kernels,
                                           top=[dict(op=e.key, device_ms=e.self_device_time_total / 10 / 1e3, share=e.self_device_time_total / 10 / max(dev, 1), calls=e.count / 10) for e in top])
            print(json.dumps(dict(time_kst=now(), profile=f'{sensor}_{size}', device_ms=dev / 1e3, host_ms=cpu / 1e3, kernels=kernels)), flush=True)
    write('profile.json', dict(time_kst=now(), setting='official', note='self device/host time per forward (mean of 10), torch.profiler', result=res)); return res


def systems(sensor, graph=False):
    l1, k2 = load('c2ms_sp', sensor, CFG['latency_seed']), load('lite', sensor, CFG['latency_seed'])
    n1, n2 = (Graphed(l1), Graphed(k2)) if graph else (l1, k2)
    return {'C2MS': n1, 'C2MS-SP (79 projection)': Projected(n1, sensor), 'C2MS-SP (85 projection)': FastProjected(n1, sensor),
            'C2MS-SP (86 projection)': FastProjectedV2(n1, sensor), 'Lite K2 + 86 projection': FastProjectedV2(n2, sensor)}


def measure(name, rounds=3, warmup=5, repeats=50):
    s = setting(name); res = {}
    for sensor in CFG['latency_sensors']:
        sys_ = systems(sensor, s['cuda_graph'])
        for size, x in inputs(sensor).items():
            per = {n: [] for n in sys_}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=s['fp16']):
                for m in sys_.values():
                    for _ in range(warmup): m(*x)
                torch.cuda.synchronize()
                for _ in range(rounds):
                    t = {n: [] for n in sys_}
                    for _ in range(repeats):
                        for n, m in sys_.items():
                            a = time.perf_counter(); m(*x); torch.cuda.synchronize(); t[n].append(time.perf_counter() - a)
                    for n in sys_: per[n].append(float(np.median(t[n])))
            res[f'{sensor}_{size}'] = {n: dict(seconds=float(np.median(v)), rounds=v) for n, v in per.items()}
            print(json.dumps(dict(time_kst=now(), timing=name, sensor=sensor, size=size, ms={n: round(1e3 * float(np.median(v)), 2) for n, v in per.items()})), flush=True)
    return res


def timing():
    out = dict(time_kst=now(), protocol=CFG['latency'], gpu=torch.cuda.get_device_name(0), settings={})
    for name in CFG['settings']:
        try: out['settings'][name] = dict(config=CFG['settings'][name], result=measure(name))
        except Exception as e: out['settings'][name] = dict(config=CFG['settings'][name], error=repr(e)); print('TIMING FAILED', name, repr(e), flush=True)
    write('timing.json', out); return out


def accuracy():
    res = {}
    for sensor in CFG['sensors']:
        res[sensor] = {}
        for name in CFG['settings']:
            s = setting(name); net = load('c2ms_sp', sensor, CFG['latency_seed']); sp = FastProjectedV2(Graphed(net) if s['cuda_graph'] else net, sensor); vals = []
            with h5py.File(Path(CFG['data_root']) / data_files(sensor)['RR']) as f:
                mv = SENSORS[sensor]['max_value']
                for i in range(len(f['pan'])):
                    a = {k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')}
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=s['fp16']):
                        vals.append(rr_metrics(a['gt'], V.predict(sp, a['pan'], a['lms'], a['ms'], mv, 'cuda')))
            res[sensor][name] = {k: float(np.mean([v[k] for v in vals])) for k in ('PSNR', 'SAM', 'ERGAS')}
        base = res[sensor]['official']; res[sensor]['change_vs_official'] = {n: {k: v[k] - base[k] for k in base} for n, v in res[sensor].items() if n in CFG['settings'] and n != 'official'}
        print(json.dumps(dict(time_kst=now(), accuracy=sensor, change=res[sensor]['change_vs_official'])), flush=True)
    write('accuracy.json', dict(time_kst=now(), system='C2MS-SP (65 + 86 projection), seed 42, RR test 20 scenes', result=res)); return res


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['identity', 'profile', 'timing', 'accuracy', 'all'], required=True); a = p.parse_args()
    write('started.json', dict(time_kst=now(), mode=a.mode, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                               sources={f: sha(HERE / f) for f in ('run.py', 'prepare.py', 'config.json')}))
    for step in (['identity', 'profile', 'timing', 'accuracy'] if a.mode == 'all' else [a.mode]): globals()[step]()
    write('complete.json', dict(time_kst=now(), mode=a.mode)); print('86 DONE', flush=True)
