"""78 data helpers (no side effects at import; the 54 trainer imports this file as `prepare`).

1. Re-exports the sensor table, Wald operators and 2x coarser pairs of experiment 54 unchanged (QB entry; the 53 construction), as 77.
2. Teacher cache: build_teacher() writes the mean output of the six fixed teacher models (53 C2MS + 65 C2MS_L1S, QB seeds 42-44)
   for the training and validation patches at the original and the 2x coarser scale (FP32). teacher_metrics() scores a cached
   teacher against GT with the 53/54 validation accumulation; run.py compares it with the stage 0 result.
3. Stage 0 teacher check (local, training-free): python prepare.py stage0 <val|train> [--limit N]
   Reads data/, 53/65 best.pt, 53 CoarsePairs and the 54 model; writes results/teacher_check_<part>.json. The decision rule was fixed
   in config.json (stage0) before the training-patch part was run. Inference on Apple MPS FP32.
"""
import os, sys, json, time, argparse, hashlib, importlib.util
from pathlib import Path
import numpy as np, h5py, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PERF = REPO / 'experiments/performance'
OUT = HERE / 'results'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_P54 = module('exp54_prepare', PERF / '54_multi_sensor_scale_mixed_training/prepare.py')
SENSORS, PHASE, SIZE = _P54.SENSORS, _P54.PHASE, _P54.SIZE
data_files, gen_mtf, Wald, CoarsePairs = _P54.data_files, _P54.gen_mtf, _P54.Wald, _P54.CoarsePairs
TEACHER_FAMILIES = ('C2MS', 'L1S')


# ---------------------------------------------------------------- teacher cache (server prepare)

def teacher_paths(cfg):
    return [REPO / cfg['teacher']['checkpoints'][fam].replace('<seed>', str(s)) for fam in TEACHER_FAMILIES for s in cfg['teacher']['seeds']]


def cache_name(split, coarse, key):
    return f"{'coarse_' if coarse else ''}{split}_{key}.npy"


@torch.inference_mode()
def build_teacher(cfg, sensor, cache, device, micro=64):
    """Write <coarse_>{train,val}_teacher.npy next to the input caches; returns {name: sha256}."""
    from model import Teacher
    teacher = Teacher(teacher_paths(cfg), SENSORS[sensor]['bands']).to(device).eval(); hashes = {}
    for split in ('train', 'val'):
        for coarse in (False, True):
            src = {k: np.load(cache / cache_name(split, coarse, k), mmap_mode='r') for k in ('pan', 'lms', 'ms', 'gt')}
            n = len(src['pan']); dest = cache / cache_name(split, coarse, 'teacher'); temp = dest.with_name(dest.stem + '_building.npy')
            arr = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=src['gt'].shape)
            for i in range(0, n, micro):
                x = [torch.from_numpy(np.array(src[k][i:i + micro])).to(device) for k in ('pan', 'lms', 'ms')]
                y = teacher(*x).float().cpu().numpy()
                if not np.isfinite(y).all(): raise FloatingPointError('teacher nonfinite')
                arr[i:i + len(y)] = y
            arr.flush(); del arr; os.replace(temp, dest); hashes[dest.name] = sha(dest)
            print('TEACHER', sensor, dest.name, flush=True)
    return hashes


def teacher_metrics(cache, split, coarse, chunk=256):
    """PSNR (mean over patches of the band-mean PSNR), SAM (pixel mean) and MSE of the cached teacher, as the 53/54 validation()."""
    t = np.load(cache / cache_name(split, coarse, 'teacher'), mmap_mode='r'); g = np.load(cache / cache_name(split, coarse, 'gt'), mmap_mode='r')
    acc = np.zeros(3); el = px = n = 0
    for i in range(0, len(g), chunk):
        p = torch.from_numpy(np.array(t[i:i + chunk])).double(); q = torch.from_numpy(np.array(g[i:i + chunk])).double(); err = (p - q).square()
        acc[0] += err.sum().item(); acc[1] += (-10 * torch.log10(err.mean((2, 3)).clamp_min(1e-15))).mean(1).sum().item()
        norm = p.norm(dim=1) * q.norm(dim=1); mask = norm > 1e-12; cos = (p * q).sum(1) / norm.clamp_min(1e-12)
        acc[2] += torch.acos(cos.clamp(-1, 1))[mask].sum().item(); px += int(mask.sum()); el += q.numel(); n += len(q)
    return dict(mse=acc[0] / el, psnr=acc[1] / n, sam=float(np.degrees(acc[2] / max(px, 1))))


# ---------------------------------------------------------------- stage 0 (local teacher check)

SEEDS = (42, 43, 44)
CKPT = {'C2MS': PERF / '53_scale_mixed_training/checkpoints/C2MS_QB_s{}/best.pt',
        'L1S': PERF / '65_l1_spectral_loss/checkpoints/C2MS_L1S_QB_s{}/best.pt'}
DATA = {'train': (REPO / 'data/train/train_qb.h5', '7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371', 17139),
        'val': (REPO / 'data/validation/valid_qb.h5', 'b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470', 1905)}
M54 = P53 = DEV = None


def stage0_setup():
    """Stage 0 environment (kept out of import so the trainer is unaffected): no grad, 4 threads, 53 CoarsePairs, MPS if present."""
    global M54, P53, DEV
    sys.path.insert(0, str(REPO)); OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    M54 = module('exp54_model', PERF / '54_multi_sensor_scale_mixed_training/model.py')
    P53 = module('exp53_prepare', PERF / '53_scale_mixed_training/prepare.py')
    DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def load():
    nets, info = {}, {}
    for fam, pat in CKPT.items():
        for s in SEEDS:
            ck = torch.load(str(pat).format(s), map_location='cpu', weights_only=False)
            n = M54.Model('C2MS', 4, s); n.load_state_dict(ck['model'], strict=True); n.eval().to(DEV)
            nets[f'{fam}_s{s}'] = n; info[f'{fam}_s{s}'] = dict(best_epoch=ck.get('best_epoch'), metrics=ck.get('metrics'))
    return nets, info


MEMBERS = [f'{f}_s{s}' for f in CKPT for s in SEEDS]
GROUPS = {'E_C2MS': [f'C2MS_s{s}' for s in SEEDS], 'E_L1S': [f'L1S_s{s}' for s in SEEDS]}
GROUPS['E_ALL'] = GROUPS['E_C2MS'] + GROUPS['E_L1S']
NAMES = MEMBERS + list(GROUPS)


class Acc:
    """Same accumulation as the 53/54 trainer validation(): MSE over elements, mean of per-patch band-mean PSNR, SAM over pixels."""
    def __init__(self):
        self.a = {k: np.zeros(3) for k in NAMES}; self.px = {k: 0 for k in NAMES}; self.el = 0; self.n = 0
        # pull of the teacher toward GT for each member: sum <t-y, g-y>, sum |g-y|^2, sum |t-y|^2
        self.pull = {g: {m: np.zeros(3) for m in mem} for g, mem in GROUPS.items()}

    def add(self, out, gt):
        for k in NAMES:
            p = out[k].double(); g = gt.double(); err = (p - g).square()
            self.a[k][0] += err.sum().item()
            self.a[k][1] += (-10 * torch.log10(err.mean((2, 3)).clamp_min(1e-15))).mean(1).sum().item()
            norm = p.norm(dim=1) * g.norm(dim=1); mask = norm > 1e-12; cos = (p * g).sum(1) / norm.clamp_min(1e-12)
            self.a[k][2] += torch.acos(cos.clamp(-1, 1))[mask].sum().item(); self.px[k] += int(mask.sum())
        for grp, mem in GROUPS.items():
            t = out[grp].double()
            for m in mem:
                y = out[m].double(); d = t - y; e = gt.double() - y
                self.pull[grp][m] += [(d * e).sum().item(), e.square().sum().item(), d.square().sum().item()]
        self.el += gt.numel(); self.n += len(gt)

    def result(self):
        met = {k: dict(mse=self.a[k][0] / self.el, psnr=self.a[k][1] / self.n, sam=float(np.degrees(self.a[k][2] / max(self.px[k], 1)))) for k in NAMES}
        gains = {}
        for grp, mem in GROUPS.items():
            mp = np.mean([met[m]['psnr'] for m in mem]); ms = np.mean([met[m]['sam'] for m in mem]); mm = np.mean([met[m]['mse'] for m in mem])
            pulls = {m: self.pull[grp][m][0] / self.pull[grp][m][1] for m in mem}
            gains[grp] = dict(d_psnr=met[grp]['psnr'] - mp, d_sam=met[grp]['sam'] - ms, mse_ratio=met[grp]['mse'] / mm,
                              pull_toward_gt=pulls, pull_mean=float(np.mean(list(pulls.values()))),
                              teacher_member_rms_over_member_error_rms={m: float(np.sqrt(self.pull[grp][m][2] / self.pull[grp][m][1])) for m in mem})
        return dict(metrics=met, gains=gains, patches=self.n)


def run(part, limit=None, chunk=256, micro=64):
    path, want, count = DATA[part]
    t0 = time.time(); digest = sha(path)
    if digest != want: raise ValueError('sha mismatch ' + part)
    nets, info = load(); coarse = P53.CoarsePairs('cpu')
    acc = {'original': Acc(), 'coarse': Acc()}
    with h5py.File(path, 'r') as f:
        n = len(f['pan']); assert n == count
        if limit: n = min(n, limit)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            x = {k: torch.from_numpy(np.asarray(f[k][i:j], dtype=np.float32) / 2047.) for k in ('pan', 'lms', 'ms', 'gt')}
            parts = [coarse(x['pan'][a:a + 8], x['gt'][a:a + 8]) for a in range(0, j - i, 8)]  # exact 53 CoarsePairs, small batches (im2col memory)
            xc = dict(zip(('pan', 'lms', 'ms', 'gt'), (torch.cat([p[q] for p in parts]) for q in range(4))))
            for scale, d in (('original', x), ('coarse', xc)):
                for a in range(0, j - i, micro):
                    inp = [d[k][a:a + micro].to(DEV) for k in ('pan', 'lms', 'ms')]
                    out = {m: nets[m](*inp).float().cpu() for m in MEMBERS}
                    for g, mem in GROUPS.items(): out[g] = torch.stack([out[m] for m in mem]).mean(0)
                    acc[scale].add(out, d['gt'][a:a + micro])
            el = time.time() - t0
            print(f'{part} {j}/{n} {el:.0f}s eta {el / j * (n - j):.0f}s', flush=True)
    res = {s: a.result() for s, a in acc.items()}
    res['checkpoint_info'] = info; res['data_sha256'] = digest; res['device'] = str(DEV); res['torch'] = torch.__version__
    res['seconds'] = round(time.time() - t0, 1); res['limit'] = limit
    if part == 'val':
        res['reproduction_vs_checkpoint'] = {m: dict(d_psnr=res['original']['metrics'][m]['psnr'] - info[m]['metrics']['val_psnr'],
                                                     d_sam=res['original']['metrics'][m]['sam'] - info[m]['metrics']['val_sam']) for m in MEMBERS}
    (OUT / f'teacher_check_{part}{"_limit" + str(limit) if limit else ""}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('stage0',)); ap.add_argument('part', choices=('val', 'train')); ap.add_argument('--limit', type=int)
    a = ap.parse_args(); stage0_setup(); r = run(a.part, a.limit)
    for s in ('original', 'coarse'):
        print(s, json.dumps({g: {k: v for k, v in d.items() if k in ('d_psnr', 'd_sam', 'mse_ratio', 'pull_mean')} for g, d in r[s]['gains'].items()}))
