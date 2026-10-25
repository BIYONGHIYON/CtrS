"""61 phase 1 worker: train one high-band head on the frozen C2MS output, or evaluate it.

The C2MS outputs of the training/validation patches (original and 2x coarser pairs) are precomputed once in FP32 by run.py
(cache y_*.npy), so a step only runs the head. Loss = MSE + lambda x angle on the original pair + the same on the coarse pair
(scale mixing kept). Epoch 0 validation is the C2MS output itself (zero-initialized head). Selection (fixed in config.json):
minimum validation SAM among epochs whose validation PSNR >= epoch-0 PSNR - 0.02 dB; epoch 0 if none qualifies.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib, json, math, random, signal, sys, time, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol  # noqa: E402
from common.metrics import fr_metrics, rr_metrics  # noqa: E402
import model as M  # noqa: E402

KST = timezone(timedelta(hours=9)); KEYS = ('pan', 'lms', 'ms', 'gt')


def now(): return datetime.now(KST).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    for attempt in range(40):
        try: os.replace(t, p); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.05)


def atomic(p, obj):
    p = Path(p); tmp = p.with_suffix('.tmp'); torch.save(obj, tmp); os.replace(tmp, p)


def setup(seed):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_name(sensor, arm, lam, seed): return f'{sensor}_{arm}_l{lam:g}_s{seed}'


def cache(cfg, sensor, seed, split, coarse=False):
    d = Path(cfg['runtime']) / 'cache' / sensor; pre = 'coarse_' if coarse else ''
    c = {k: np.load(d / f'{pre}{split}_{k}.npy', mmap_mode='r') for k in ('pan', 'lms', 'gt')}
    c['y'] = np.load(d / f'{pre}{split}_y_s{seed}.npy', mmap_mode='r'); return c


def batch(c, idx):
    return [torch.from_numpy(np.array(c[k][idx], copy=True)).cuda(non_blocking=True) for k in ('y', 'pan', 'lms', 'gt')]


@torch.no_grad()
def validation(head, c, size=64):
    head.eval(); se = 0.; elements = 0; psnr = 0.; ang = 0.; pixels = 0; n = 0
    for i in range(0, len(c['y']), size):
        y, pan, lms, g = batch(c, slice(i, i + size)); p = head(y, pan, lms)
        if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite validation')
        err = (p - g).square(); se += float(err.double().sum()); elements += g.numel(); n += len(g)
        psnr += float((-10 * torch.log10(err.mean((2, 3)).clamp_min(1e-15))).mean(1).double().sum())
        norm = p.norm(dim=1) * g.norm(dim=1); mask = norm > 1e-12; cos = (p * g).sum(1) / norm.clamp_min(1e-12)
        ang += float(torch.acos(cos.clamp(-1, 1))[mask].double().sum()); pixels += int(mask.sum())
    return dict(val_mse=se / elements, val_psnr=psnr / n, val_sam=math.degrees(ang / max(pixels, 1)))


def selected(history, tol):
    """Selection rule fixed in config.json: min val SAM among epochs with val PSNR >= epoch-0 PSNR - tol; else epoch 0."""
    floor = history[0]['val_psnr'] - tol
    ok = [h for h in history if h['val_psnr'] >= floor]
    return min(ok, key=lambda h: (h['val_sam'], h['epoch']))['epoch']


def train(cfg, sensor, arm, lam, seed, resume=False):
    setup(seed); out = Path(cfg['runtime']) / 'runs' / run_name(sensor, arm, lam, seed); cp = out / 'checkpoints'; cp.mkdir(parents=True, exist_ok=True)
    if (out / 'training_complete.json').exists(): return
    if (cp / 'latest.pt').exists() and not resume: raise RuntimeError('existing run; explicit --resume required')
    head = M.Head(arm, sensor, seed).cuda(); opt = torch.optim.Adam(head.parameters(), lr=cfg['lr_start'])
    tr, va = cache(cfg, sensor, seed, 'train'), cache(cfg, sensor, seed, 'val')
    trc, vac = cache(cfg, sensor, seed, 'train', True), cache(cfg, sensor, seed, 'val', True)
    n = len(tr['y']); steps = math.ceil(n / cfg['batch']); E = cfg['epochs']; cw = cfg['coarse_weight']; tol = cfg['selection_psnr_tolerance_db']
    gen = torch.Generator().manual_seed(seed + 100000); epoch = 0; history = []
    stop = [False]; signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0, True))
    def log(phase, **kw):
        rec = dict(time_kst=now(), pid=os.getpid(), run=out.name, phase=phase, **kw); write(out / 'status.json', rec)
        with (out / 'events.jsonl').open('a', encoding='utf8') as f: f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
    if resume and (cp / 'latest.pt').exists():
        ck = torch.load(cp / 'latest.pt', map_location='cpu', weights_only=False)
        if ck['fingerprint'] != cfg['fingerprint']: raise RuntimeError('resume contract mismatch')
        head.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); gen.set_state(ck['generator'])
        torch.set_rng_state(ck['cpu_rng']); torch.cuda.set_rng_state(ck['cuda_rng']); epoch = ck['epoch']; history = ck['history']
    else:
        v = validation(head, va); vc = validation(head, vac)
        history = [dict(epoch=0, train_loss=None, **v, **{'coarse_' + k: x for k, x in vc.items()})]
        atomic(cp / 'best.pt', dict(model=head.state_dict(), epoch=0, metrics=history[0], sensor=sensor, arm=arm, lam=lam, seed=seed, fingerprint=cfg['fingerprint']))
        write(out / 'initialization.json', dict(parameters=sum(p.numel() for p in head.parameters()), **history[0])); log('epoch0_validation', **history[0])
    while epoch < E:
        start = time.monotonic(); head.train()
        lr = cfg['lr_end'] + .5 * (cfg['lr_start'] - cfg['lr_end']) * (1 + math.cos(math.pi * epoch / (E - 1)))
        for g in opt.param_groups: g['lr'] = lr
        order = torch.randperm(n, generator=gen); total = torch.zeros((), device='cuda', dtype=torch.float64)
        for bi in range(steps):
            idx = order[bi * cfg['batch']:(bi + 1) * cfg['batch']].sort().values.numpy()
            y, pan, lms, g = batch(tr, idx); p = head(y, pan, lms)
            loss = (p - g).square().mean() + (lam * M.angle(p, g) if lam else 0.)
            if cw:
                yc, panc, lmsc, gc = batch(trc, idx); pc = head(yc, panc, lmsc)
                loss = loss + cw * ((pc - gc).square().mean() + (lam * M.angle(pc, gc) if lam else 0.))
            opt.zero_grad(set_to_none=True); loss.backward()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            opt.step(); total += loss.detach().double()
            if stop[0]: log('interrupted'); return
        v = validation(head, va); vc = validation(head, vac); epoch += 1
        row = dict(epoch=epoch, train_loss=float(total) / steps, lr=lr, **v, **{'coarse_' + k: x for k, x in vc.items()}, epoch_seconds=time.monotonic() - start)
        history.append(row)
        if selected(history, tol) == epoch:
            atomic(cp / 'best.pt', dict(model=head.state_dict(), epoch=epoch, metrics=row, sensor=sensor, arm=arm, lam=lam, seed=seed, fingerprint=cfg['fingerprint']))
        write(out / 'history.json', history)
        atomic(cp / 'latest.pt', dict(model=head.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(), cpu_rng=torch.get_rng_state(),
                                      cuda_rng=torch.cuda.get_rng_state(), epoch=epoch, history=history, fingerprint=cfg['fingerprint']))
        log('epoch_complete', **row, selected_epoch=selected(history, tol))
    best = selected(history, tol)
    write(out / 'training_complete.json', dict(time_kst=now(), epochs=E, selected_epoch=best, selected=history[best], epoch0=history[0],
                                               best_sha256=sha(cp / 'best.pt'), fingerprint=cfg['fingerprint']))
    log('training_complete', selected_epoch=best)


class Full(torch.nn.Module):
    def __init__(self, c2ms, head):
        super().__init__(); self.c2ms, self.head = c2ms, head

    def forward(self, pan, lms, ms):
        y = self.c2ms(pan, lms, ms)
        return y if self.head is None else self.head(y, pan, lms)


def c2ms_model(cfg, sensor, seed):
    ck = torch.load(ROOT / cfg['controls'][sensor].replace('<seed>', str(seed)), map_location='cpu', weights_only=False)
    m = M.C2MS(sensor, seed); m.load_state_dict(ck['model'], strict=True); return m.cuda().eval()


@torch.no_grad()
def evaluate(cfg, sensor, arm, lam, seed):
    """RR, RR2, FR_WALD and FR (+ observation RMSE) of the control C2MS and of C2MS + selected head on all 20 test scenes."""
    setup(seed); out = Path(cfg['runtime']) / 'runs' / run_name(sensor, arm, lam, seed); done = out / 'evaluation_complete.json'
    if done.exists(): return
    ck = torch.load(out / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    head = M.Head(arm, sensor, seed); head.load_state_dict(ck['model'], strict=True); c2ms = c2ms_model(cfg, sensor, seed)
    nets = {'C2MS': Full(c2ms, None).eval(), 'candidate': Full(c2ms, head.cuda().eval()).eval()}
    result = dict(time_kst=now(), run=out.name, selected_epoch=ck['epoch'], best_sha256=sha(out / 'checkpoints/best.pt'), targets={})
    for target in [sensor] + (['WV2'] if sensor == 'WV3' else []):
        wald = M.P54.Wald(target, 'cuda'); s = wald.max_value; files = M.P54.data_files(target); scenes = {}
        for proto in ('RR', 'FR'):
            with h5py.File(Path(cfg['data_root']) / files[proto]) as f:
                scenes[proto] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in KEYS if k in f} for i in range(len(f['pan']))]
        def predict(net, pan, lms, ms):
            x = [torch.from_numpy((np.asarray(a, dtype=np.float64) / s).astype('float32')).unsqueeze(0).cuda() for a in (pan, lms, ms)]
            y = net(*x)[0].float().cpu().numpy().astype(np.float64) * s
            if not np.isfinite(y).all(): raise FloatingPointError('nonfinite prediction')
            return y
        rows = {k: [] for k in nets}
        for rr, fr in zip(scenes['RR'], scenes['FR']):
            for key, net in nets.items():
                r = {'RR': rr_metrics(rr['gt'], predict(net, rr['pan'], rr['lms'], rr['ms'])),
                     'RR2': rr_metrics(rr['ms'], predict(net, *wald.degrade(rr['pan'], rr['ms'], 'cuda'))),
                     'FR_WALD': rr_metrics(fr['ms'], predict(net, *wald.degrade(fr['pan'], fr['ms'], 'cuda')))}
                fused = predict(net, fr['pan'], fr['lms'], fr['ms']); r['FR'] = fr_metrics(fused, fr['lms'], fr['ms'], fr['pan'])
                r['FR']['OBS_RMSE_DN'] = wald.observation_rmse(fused, fr['ms'], 'cuda', protocol.OBS_MARGIN); rows[key].append(r)
        mean = lambda rs: {part: {k: float(np.mean([x[part][k] for x in rs])) for k in rs[0][part]} for part in rs[0]}
        result['targets'][target] = {k: dict(mean=mean(v), scenes=v) for k, v in rows.items()}
        print(json.dumps(dict(run=out.name, target=target, C2MS=result['targets'][target]['C2MS']['mean']['RR'],
                              candidate=result['targets'][target]['candidate']['mean']['RR'])), flush=True)
    write(done, result)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    p.add_argument('--sensor', required=True); p.add_argument('--arm', choices=M.ARMS, required=True); p.add_argument('--lam', type=float, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding='utf8'))
    try:
        if a.mode == 'train': train(cfg, a.sensor, a.arm, a.lam, a.seed, a.resume)
        else: evaluate(cfg, a.sensor, a.arm, a.lam, a.seed)
    except BaseException:
        folder = Path(cfg['runtime']) / 'runs' / run_name(a.sensor, a.arm, a.lam, a.seed); folder.mkdir(parents=True, exist_ok=True)
        write(folder / f'failure_{a.mode}.json', dict(time_kst=now(), traceback=traceback.format_exc())); raise
