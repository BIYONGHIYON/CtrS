"""D2 (training-free, post-hoc): test-time per-band radiometric calibration to the training sensor, 3 seeds, three transfer pairs and in-domain.
ratio : each MS band scaled per scene so that mean(band)/mean(PAN) equals the training sensor's reference ratio (PAN untouched); output scaled back.
level : PAN and every band scaled per scene so that the channel means equal the training sensor's reference means; output scaled back.
ratio_const : like ratio, but one constant per band for the whole target test set (no per-scene adaptation)."""
import sys, json, time, importlib.util
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parents[5]; PERF = ROOT / 'experiments/performance'
spec = importlib.util.spec_from_file_location('p66', PERF / '66_observation_error_projection_analysis/prepare.py')
p66 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p66)
from common.metrics import rr_metrics
torch.set_num_threads(7); torch.set_grad_enabled(False)
RS = json.loads(Path(sys.argv[2]).read_text())
MAXV = {'QB': 2047., 'GF2': 1023., 'WV3': 2047., 'WV2': 2047.}; BANDS = {'QB': 4, 'GF2': 4, 'WV3': 8, 'WV2': 8}
def ref(train):
    if train == 'QB': r = RS['QB_train']; return dict(pan=r['pan'], bands=np.array(r['bands']), ratio=np.array(r['ratio_of_means']))
    r = RS[f'{train}_test']; return dict(pan=r['pan'], bands=np.array(r['bands']), ratio=np.array(r['scene_ratio_mean']))

def detail(y, gt, lms):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    high = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) * f)); hy, hg = high(y), high(gt)
    return dict(detail_energy=float(np.sqrt((hy ** 2).mean() / (hg ** 2).mean())), injected_ratio=float((y - lms).std() / (gt - lms).std()), band_bias_dn=float(np.abs((y - gt).mean((1, 2))).mean()),
                detail_corr=float((hy * hg).sum() / np.sqrt((hy ** 2).sum() * (hg ** 2).sum())))

def run(train, target, seeds, conds, proj_conds, K, res, out):
    p66.use(target); St = MAXV[target]; Str = MAXV[train]; R = ref(train); tag = p66.SENSORS[target]['tag']
    with h5py.File(ROOT / f'data/test/test_{tag}_multiExm1.h5') as f:
        data = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
    tgt_ratio = np.mean([x['lms'].mean((1, 2)) / x['pan'].mean() for x in data], 0)
    def scales(cond, x):
        """(pan input factor, per-band input factor, per-band output factor): net inputs = DN * factor, output DN = net output * out factor."""
        mp = x['pan'].mean(); mb = x['lms'].mean((1, 2))
        if cond == 'raw': t = np.ones_like(mb); return 1 / St, t / St, St / t
        if cond == 'ratio': t = R['ratio'] * mp / mb; return 1 / St, t / St, St / t
        if cond == 'ratio_const': t = R['ratio'] / tgt_ratio; return 1 / St, t / St, St / t
        if cond == 'level': kp = (R['pan'] / Str) / mp; kb = (R['bands'] / Str) / mb; return kp, kb, 1 / kb
        raise ValueError(cond)
    for seed in seeds:
        for name, path in p66.models(seed, train).items():
            if name not in ('C2MS', 'L1S'): continue
            m = p66.load_model(path, BANDS[train])
            for cond in conds:
                for proj in ((False, True) if cond in proj_conds else (False,)):
                    rows = []; t0 = time.time()
                    for x in data:
                        kp, kb, ko = scales(cond, x); kb3, ko3 = kb[:, None, None], ko[:, None, None]
                        tin = [torch.from_numpy(a.astype('float32'))[None] for a in (x['pan'] * kp, x['lms'] * kb3, x['ms'] * kb3)]
                        y = m(*tin).double()[0].numpy() * ko3
                        if proj:
                            yy, _ = p66.project(torch.from_numpy(y / St)[None], torch.from_numpy(x['ms'] / St)[None], max_iter=K); y = yy[0].numpy() * St
                        r = rr_metrics(x['gt'], y); r = {k: float(r[k]) for k in ('PSNR', 'SAM', 'ERGAS')}; r.update(detail(y, x['gt'], x['lms'])); rows.append(r)
                    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                    key = f'{train}->{target}|{name}|s{seed}|{cond}|{"P" if proj else "-"}'; res[key] = dict(mean=mean, rows=rows)
                    print(key, {k: round(v, 4) for k, v in mean.items()}, f'{time.time() - t0:.0f}s', flush=True)
                    Path(out).write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    # usage: out ref_stats threads "TRAIN>TARGET:seeds:conds:projconds;..."
    out = sys.argv[1]; res = {}; K = 10; torch.set_num_threads(int(sys.argv[3]))
    for item in sys.argv[4].split(';'):
        pair, seeds, conds, pc = item.split(':'); train, target = pair.split('>')
        run(train, target, [int(x) for x in seeds.split(',')], conds.split(','), [c for c in pc.split(',') if c], K, res, out)
