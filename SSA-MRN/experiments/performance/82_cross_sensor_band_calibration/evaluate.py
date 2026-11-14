"""82 evaluation (no training; local CPU or MPS): archived models applied to a sensor they were not trained on, with and without the per-band
sensor calibration of prepare.py, on the 20 RR test scenes of the target (FR QNR for the main conditions, reference).

Pairs (training sensor -> target): WV3 -> WV2, QB -> GF2, GF2 -> QB. Models: 54/53 C2MS and 65 C2MS_L1S, seeds 42-44; seed-42 references on
WV3 -> WV2: 59 C2MS 300 epochs, 67 C2MS_B and L1S_GB, 71 BLUR_COARSE. Conditions: raw, raw+P, cal, cal+P (main), const+P, level+P.
In-domain guard check (why calibration is never applied to the training sensor): QB, GF2, WV3 with 'cal' without projection.
Extra detail measures per scene (as 67): detail_energy = sqrt(E|high(y)|^2 / E|high(gt)|^2) above the MS Nyquist (1/8 cycles/pixel),
detail_corr = correlation of high(y) and high(gt), injected_ratio = std(y - LMS) / std(gt - LMS), band_bias_dn = mean over bands of |mean(y - gt)|.
Inputs are normalized by the target sensor's maximum value (the 54 transfer convention). All of this is reference: under criteria v2.4 a
transfer test is not judged."""
import importlib.util, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from prepare import SENSORS, data_files, Projected, Calibrated
from common.metrics import rr_metrics, fr_metrics
from common import result_images
_spec = importlib.util.spec_from_file_location('exp83_model', HERE.parent / '83_factorized_c2ms_finetuning/model.py')
M83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M83); M54 = M83.M54
torch.set_grad_enabled(False)


def detail(y, gt, lms):
    h, w = gt.shape[-2:]; f = np.maximum(np.abs(np.fft.fftfreq(h))[:, None], np.abs(np.fft.fftfreq(w))[None, :]) >= 1 / 8
    hi = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) * f)); hy, hg = hi(y), hi(gt)
    return dict(detail_energy=float(np.sqrt((hy ** 2).mean() / (hg ** 2).mean())), detail_corr=float((hy * hg).sum() / np.sqrt((hy ** 2).sum() * (hg ** 2).sum())),
                injected_ratio=float((y - lms).std() / (gt - lms).std()), band_bias_dn=float(np.abs((y - gt).mean((1, 2))).mean()))


def predict(net, a, scale, device):
    x = [torch.from_numpy((a[k] / scale).astype('float32'))[None].to(device) for k in ('pan', 'lms', 'ms')]
    y = net(*x)[0].double().cpu().numpy() * scale
    if not np.isfinite(y).all(): raise FloatingPointError('nonfinite')
    return y


def load(path, bands, seed, device):
    ck = torch.load(ROOT / path, map_location='cpu', weights_only=False)
    if ck['seed'] != seed: raise ValueError('seed mismatch ' + str(path))
    m = M54.Model('C2MS', bands, seed); m.load_state_dict(ck['model'], strict=True); return m.to(device).eval()


def main(out, device, threads, only=None):
    torch.set_num_threads(threads); cfg = json.loads((HERE / 'config.json').read_text(encoding='utf8')); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    refs = {s: json.loads((HERE / p).read_text(encoding='utf8')) for s, p in cfg['reference_stats'].items() if (HERE / p).exists()}  # GF2/WV3 arrive from the server
    res_path = out / 'evaluation.json'; res = json.loads(res_path.read_text(encoding='utf8')) if res_path.exists() else {}
    data = {}
    def test(target):
        if target not in data:
            p = ROOT / 'data/test' / Path(data_files(target)['RR']).name; q = ROOT / 'data/test' / Path(data_files(target)['FR']).name
            with h5py.File(p) as f: rr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms', 'gt')} for i in range(20)]
            with h5py.File(q) as f: fr = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')} for i in range(20)]
            data[target] = (rr, fr, np.mean([x['lms'].mean((1, 2)) / x['pan'].mean() for x in rr], 0))
        return data[target]
    def system(net, train, target, cond):
        rr, fr, tratio = test(target); ref = refs[train]
        base = {'raw': net, 'cal': Calibrated(net, ref, 'scene'), 'const': Calibrated(net, ref, 'const', target_ratio=tratio),
                'level': Calibrated(net, ref, 'level', scales=(SENSORS[train]['max_value'], SENSORS[target]['max_value'], ref['pan_mean_dn'], ref['band_mean_dn']))}[cond.split('+')[0]]
        return Projected(base, target) if cond.endswith('+P') else base
    def run(key, net, train, target, cond, with_fr):
        if key in res: return
        t0 = time.time(); rr, fr, _ = test(target); s = SENSORS[target]['max_value']; m = system(net, train, target, cond); rows = []
        for a in rr:
            y = predict(m, a, s, device); r = rr_metrics(a['gt'], y); r = {k: float(v) for k, v in r.items()}; r.update(detail(y, a['gt'], a['lms'])); rows.append(r)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if with_fr:
            frs = [fr_metrics(predict(m, b, s, device), b['lms'], b['ms'], b['pan']) for b in fr]; mean.update({'FR_' + k: float(np.mean([x[k] for x in frs])) for k in frs[0]})
        res[key] = dict(train=train, target=target, condition=cond, mean=mean, rows=rows, seconds=round(time.time() - t0, 1))
        res_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf8')
        print(key, {k: round(v, 4) for k, v in mean.items() if k in ('PSNR', 'SAM', 'ERGAS', 'detail_energy', 'detail_corr', 'band_bias_dn', 'FR_QNR')}, f'{time.time() - t0:.0f}s', flush=True)
    for pair in cfg['pairs']:
        train, target = pair.split('>')
        if only and pair not in only: continue
        for seed in cfg['seeds']:
            for name, paths in cfg['models'].items():
                net = load(paths[train].replace('<seed>', str(seed)), SENSORS[train]['bands'], seed, device)
                for cond in cfg['conditions']: run(f'{pair}|{name}|s{seed}|{cond}', net, train, target, cond, cond in cfg['fr_conditions'])
        if pair == 'WV3>WV2':
            for name, path in cfg['references_seed42'].items():
                net = load(path, 8, 42, device)
                for cond in ('raw+P', 'cal+P'): run(f'{pair}|{name}|s42|{cond}', net, train, target, cond, False)
    for sensor in cfg['in_domain_guard']:
        if only and 'guard' not in only: continue
        for seed in cfg['seeds']:
            for name, paths in cfg['models'].items():
                net = load(paths[sensor].replace('<seed>', str(seed)), SENSORS[sensor]['bands'], seed, device)
                for cond in ('raw', 'cal'): run(f'{sensor}>{sensor}|{name}|s{seed}|{cond}', net, sensor, sensor, cond, False)
    if not only or 'images' in only:
        net = load(cfg['models']['L1S']['WV3'].replace('<seed>', '42'), 8, 42, device); c2 = load(cfg['models']['C2MS']['WV3'].replace('<seed>', '42'), 8, 42, device)
        for tag, base, cand, names, labels in (('C2MS_vs_L1S_cal_P', c2, system(net, 'WV3', 'WV2', 'cal+P'), ('C2MS', 'L1S+cal+P'), ('현재 기준 C2MS (보정 없음)', '65번 + 밴드 보정 + 사영')),
                                              ('L1S_P_vs_L1S_cal_P', Projected(net, 'WV2'), system(net, 'WV3', 'WV2', 'cal+P'), ('L1S+P', 'L1S+cal+P'), ('65번 + 사영 (보정 없음)', '65번 + 밴드 보정 + 사영'))):
            result_images.render(base, cand, ROOT / 'data/test', out / 'images' / tag, names=names, seed=42, device=device, labels=labels,
                                 rr_file='test_wv2_multiExm1.h5', scale=2047., bands=SENSORS['WV2']['rgb'], land_cover={i: f'WV2 장면 {i}' for i in result_images.SCENES})
        print('images done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'cpu', int(sys.argv[3]) if len(sys.argv) > 3 else 7, sys.argv[4].split(',') if len(sys.argv) > 4 else None)
