"""Experiment 54 evaluation (no training): criteria v2.3 judgements per sensor, result images and graphs.

Measures C0 (original SSA-MRN) and C2MS (current baseline) trained on GF2 and WV3 at seeds 42-44 on all 20 RR and FR
test scenes of their own sensor, and the WV3-trained models also on WV2 (transfer). Metrics follow
common/evaluation_protocol.scene_metrics (RR, RR2, FR_WALD, FR with OBS_RMSE_DN) with the sensor's own Wald operator
from prepare.py (MS and PAN MTF, x4, phase 2) and inputs normalized by the sensor maximum (GF2 1023, WV3/WV2 2047).
Judgement: C2MS vs C0 of the same seeds per target (config.json judgement), seed means, v2.3 thresholds unchanged.
Also checks the operators against the test data and the QB operator against common/evaluation_protocol, and the measured
RR means against the controller's own evaluation records. Writes only below results/ (or --out).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, json, sys
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol, result_images  # noqa: E402
from common.interpolation import interp23tap  # noqa: E402
from common.metrics import fr_metrics, rr_metrics  # noqa: E402
sys.path.insert(0, str(HERE))
from model import Model  # noqa: E402
from prepare import SENSORS, Wald  # noqa: E402

CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
ARMS = ('C0', 'C2MS')
ALL = tuple(CFG['seeds'])
TRAINED = tuple(CFG['sensors'])
TARGETS = {'GF2': 'GF2', 'WV3': 'WV3', 'WV2': 'WV3'}  # test target -> sensor the models were trained on
NOISE = {1: 0.06, 2: 0.04}  # docs/judgement_criteria.md v2.3: RR PSNR noise threshold for 1 seed / 2 or more seeds
TOL = {'RR2': (-0.05, 0.06), 'FR_WALD': (-0.07, 0.04), 'FR_OBS': 1.10}
COLORS = {'C0': '#000000', 'C2MS': '#C44E52'}
LABEL_EN = {'개선': 'improved', '부분 개선': 'partially improved', '개선 없음': 'not improved'}  # graph titles (matplotlib default font has no Hangul)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def tag(sensor): return SENSORS[sensor]['tag']
def files(sensor): return {'RR': f'test_{tag(sensor)}_multiExm1.h5', 'FR': f'test_{tag(sensor)}_OrigScale_multiExm1.h5'}
def run_dir(runtime, arm, sensor, seed): return runtime / 'runs' / f'{sensor}_{arm}_s{seed}'


def expected_sha(sensor, proto):
    table = CFG['data_sha256'] if sensor in CFG['data_sha256'] else CFG['transfer_data_sha256']
    return table[sensor][proto]


def predict(net, pan, lms, ms, scale, device):
    x = [torch.from_numpy((np.asarray(a, dtype=np.float64) / scale).astype('float32')).unsqueeze(0).to(device) for a in (pan, lms, ms)]
    with torch.no_grad():
        y = net(*x)
    out = y[0].float().cpu().numpy().astype(np.float64) * scale
    if not np.isfinite(out).all(): raise FloatingPointError('nonfinite prediction')
    return out


def scene_metrics(net, rr, fr, wald, device):
    """protocol.scene_metrics with the target sensor's Wald operator and normalization."""
    s, out = wald.max_value, {}
    out['RR'] = rr_metrics(rr['gt'], predict(net, rr['pan'], rr['lms'], rr['ms'], s, device))
    out['RR2'] = rr_metrics(rr['ms'], predict(net, *wald.degrade(rr['pan'], rr['ms'], device), s, device))
    out['FR_WALD'] = rr_metrics(fr['ms'], predict(net, *wald.degrade(fr['pan'], fr['ms'], device), s, device))
    fused = predict(net, fr['pan'], fr['lms'], fr['ms'], s, device)
    out['FR'] = fr_metrics(fused, fr['lms'], fr['ms'], fr['pan'])
    out['FR']['OBS_RMSE_DN'] = wald.observation_rmse(fused, fr['ms'], device, protocol.OBS_MARGIN)
    return out


def judge(cand, ctrl):
    """v2.3 on seed means (lists of per-seed mean dicts); same rule as 53 evaluate.py."""
    mean = lambda rows, part, k: float(np.mean([r[part][k] for r in rows]))
    d = {part: {k: mean(cand, part, k) - mean(ctrl, part, k) for k in ctrl[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
    obs = mean(cand, 'FR', 'OBS_RMSE_DN') / mean(ctrl, 'FR', 'OBS_RMSE_DN')
    noise = NOISE[1 if len(cand) == 1 else 2]
    rr = {'PSNR': d['RR']['PSNR'] > 0, 'SAM': d['RR']['SAM'] < 0, 'ERGAS': d['RR']['ERGAS'] < 0}
    guards = {k: d[k]['PSNR'] >= TOL[k][0] and d[k]['SAM'] <= TOL[k][1] for k in ('RR2', 'FR_WALD')}
    guards['FR_OBS'] = obs <= TOL['FR_OBS']
    above = d['RR']['PSNR'] >= noise
    label = '개선' if all(rr.values()) and above and all(guards.values()) else '부분 개선' if any(rr.values()) else '개선 없음'
    return dict(delta=d, obs_ratio=obs, noise_threshold=noise, above_noise=above, rr_components_improved=rr, guards=guards, label=label)


def scene_wins(cand_rows, ctrl_rows):
    """Scenes where the candidate is better; rows are lists over seeds (scene metrics averaged over the paired seeds)."""
    avg = lambda runs: [{part: {k: float(np.mean([run[i][part][k] for run in runs])) for k in ('PSNR', 'SAM')} for part in ('RR', 'RR2', 'FR_WALD')} for i in range(len(runs[0]))]
    cand_rows, ctrl_rows = avg(cand_rows), avg(ctrl_rows)
    return {part: {'PSNR': int(sum(c[part]['PSNR'] > b[part]['PSNR'] for c, b in zip(cand_rows, ctrl_rows))),
                   'SAM': int(sum(c[part]['SAM'] < b[part]['SAM'] for c, b in zip(cand_rows, ctrl_rows)))} for part in ('RR', 'RR2', 'FR_WALD')}


def load(data, sensor, limit):
    scenes = {}
    for proto, name in files(sensor).items():
        found = list(data.rglob(name)); assert len(found) == 1, name
        digest = sha(found[0])
        if digest != expected_sha(sensor, proto): raise ValueError(f'data sha mismatch {sensor} {proto}')
        with h5py.File(found[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if proto == 'RR' else ('pan', 'lms', 'ms')
            n = len(f['pan']) if limit is None else limit
            scenes[proto] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(n)]
        scenes[proto + '_sha256'] = digest
    return scenes


def operator_check(data, sensor, scenes, device):
    """Max |DN| differences: Wald MS of the RR GT vs the given RR MS, interp23tap of the MS vs the given LMS (RR and FR)."""
    w = Wald(sensor, device); out = {'ms_from_gt': 0., 'lms_rr': 0., 'lms_fr': 0.}
    for a in scenes['RR']:
        g = torch.from_numpy(a['gt'])[None].to(device)
        out['ms_from_gt'] = max(out['ms_from_gt'], float(np.abs(w.ms(g)[0].cpu().numpy() - a['ms']).max()))
    for proto, key in (('RR', 'lms_rr'), ('FR', 'lms_fr')):
        for a in scenes[proto]:
            lms = interp23tap(torch.from_numpy(a['ms'] / w.max_value).float()[None].to(device)).double()[0].cpu().numpy() * w.max_value
            out[key] = max(out[key], float(np.abs(lms - a['lms']).max()))
    return out


def qb_operator_check(data, device):
    """prepare.Wald('QB') must equal the criteria-v2 operator of common/evaluation_protocol on a QB RR scene."""
    found = list(data.rglob('test_qb_multiExm1.h5'))
    if len(found) != 1: return 'QB test data not found'
    with h5py.File(found[0], 'r') as f: a = {k: np.asarray(f[k][0], dtype=np.float64) for k in ('pan', 'ms', 'gt')}
    ours, ref = Wald('QB', device).degrade(a['pan'], a['ms'], device), protocol.WaldDegradation('QB', device)(a['pan'], a['ms'])
    obs = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(device).double()
    g = torch.from_numpy(a['gt'])[None].to(device)
    return {'degrade_max_abs': max(float(np.abs(x - y).max()) for x, y in zip(ours, ref)),
            'observation_max_abs': float((Wald('QB', device).ms(g) - obs(g)).abs().max())}


def graphs(models, judgements, runtime, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if runtime is not None:
        fig, ax = plt.subplots(len(TRAINED), 2, figsize=(13, 4.2 * len(TRAINED))); ax = ax.ravel()
        for j, sensor in enumerate(TRAINED):
            for arm in ARMS:
                for n, s in enumerate(ALL):
                    p = run_dir(runtime, arm, sensor, s) / 'history.json'
                    if not p.exists(): continue
                    h = [r for r in json.loads(p.read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
                    kw = dict(color=COLORS[arm], alpha=1 if s == 42 else .45, lw=1.4 if s == 42 else .9, label=arm if n == 0 else None)
                    ax[2 * j].plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], **kw)
                    c = [r for r in h if r.get('coarse_val_psnr') is not None]
                    ax[2 * j + 1].plot([r['epoch'] for r in c], [r['coarse_val_psnr'] for r in c], **kw)
            ax[2 * j].set(xlabel='epoch', ylabel='PSNR (dB, peak 1)', title=f'{sensor} validation, original scale (selection)')
            ax[2 * j + 1].set(xlabel='epoch', ylabel='PSNR (dB, peak 1)', title=f'{sensor} validation, 2x coarser pairs (logged only)')
            for x in ax[2 * j:2 * j + 2]: x.set_ylim(bottom=25)
        for x in ax: x.legend(fontsize=8); x.grid(alpha=.3)
        fig.suptitle('Learning curves, seed 42 bold and seeds 43-44 light', fontsize=10)
        fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 4.2), sharey=True); tests = ('RR', 'RR2', 'FR_WALD'); x = np.arange(len(tests))
    for a, target in zip(ax, TARGETS):
        j = judgements[f'{target}:C2MS:C0@' + '+'.join(map(str, ALL))]
        per = [[judgements[f'{target}:C2MS:C0@{s}']['delta'][t]['PSNR'] for s in ALL] for t in tests]
        v = [j['delta'][t]['PSNR'] for t in tests]
        a.bar(x, v, .55, color=COLORS['C2MS'])
        for xi, d in zip(x, per): a.scatter([xi] * len(d), d, color='black', s=9, zorder=3)
        for xi, vi in zip(x, v): a.text(xi + .3, vi, f'{vi:+.2f}', va='center', fontsize=8)
        a.axhline(0, color='black', lw=.8); a.set_xticks(x, tests); a.grid(axis='y', alpha=.3)
        a.set_title(f'{target}' + (' (WV3-trained, transfer)' if target != TARGETS[target] else '') + f': {LABEL_EN[j["label"]]}', fontsize=10)
    ax[0].set_ylabel('PSNR difference C2MS - C0, same seed (dB)')
    fig.suptitle('Mean of 20 test scenes; bars = 3-seed mean, dots = individual seeds', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(CFG['runtime']), help='training runtime with runs/<sensor>_<arm>_s<seed>/checkpoints/best.pt')
    p.add_argument('--out', type=Path, default=HERE / 'results')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='first N scenes only (code check; the judgement needs all 20)')
    p.add_argument('--graphs-only', action='store_true', help='redraw graphs from <out>/evaluation_v2_3.json')
    a = p.parse_args()
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    if a.graphs_only:
        r = json.loads((out / 'evaluation_v2_3.json').read_text(encoding='utf-8'))
        return graphs(r['models'], r['judgements'], a.runtime if a.runtime.exists() else None, out)
    if (out / 'evaluation_v2_3.json').exists(): raise FileExistsError(out / 'evaluation_v2_3.json')
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    nets, models = {}, {}
    for sensor in TRAINED:
        for arm in ARMS:
            for s in ALL:
                path = run_dir(a.runtime, arm, sensor, s) / 'checkpoints/best.pt'; ck = torch.load(path, map_location='cpu', weights_only=False)
                if (ck['sensor'], ck['arm'], ck['seed']) != (sensor, arm, s): raise ValueError(f'checkpoint identity {path}')
                net = Model(arm, SENSORS[sensor]['bands'], s); net.load_state_dict(ck['model'], strict=True); nets[f'{arm}_{sensor}_s{s}'] = net.to(a.device).eval()
                models[f'{arm}_{sensor}_s{s}'] = dict(checkpoint=str(path), checkpoint_sha256=sha(path), best_epoch=ck['best_epoch'], targets={})
    checks, data_sha = {'QB_operator_vs_protocol': qb_operator_check(a.data, a.device)}, {}
    for target, sensor in TARGETS.items():
        scenes = load(a.data, target, a.limit); data_sha[target] = {k: scenes[k + '_sha256'] for k in ('RR', 'FR')}
        checks[f'{target}_operators'] = operator_check(a.data, target, scenes, a.device)
        wald = Wald(target, a.device)
        for arm in ARMS:
            for s in ALL:
                key = f'{arm}_{sensor}_s{s}'
                rows = [{'index': i, **scene_metrics(nets[key], rr, fr, wald, a.device)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
                mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
                models[key]['targets'][target] = dict(mean=mean, rows=rows)
                print(target, key, 'RR %.4f/%.4f/%.4f RR2 %.4f/%.4f FR_WALD %.4f/%.4f OBS %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'], mean['RR2']['PSNR'],
                      mean['RR2']['SAM'], mean['FR_WALD']['PSNR'], mean['FR_WALD']['SAM'], mean['FR']['OBS_RMSE_DN']), flush=True)
        del scenes
    judgements = {}
    for target, sensor in TARGETS.items():
        for seeds in (ALL,) + tuple((s,) for s in ALL):
            name = f'{target}:C2MS:C0@' + '+'.join(map(str, seeds))
            cand = [models[f'C2MS_{sensor}_s{s}']['targets'][target] for s in seeds]; ctrl = [models[f'C0_{sensor}_s{s}']['targets'][target] for s in seeds]
            j = judge([c['mean'] for c in cand], [c['mean'] for c in ctrl]); j['seeds'] = list(seeds); j['scenes_improved'] = scene_wins([c['rows'] for c in cand], [c['rows'] for c in ctrl])
            judgements[name] = j
            print(name, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d/%d' % (
                j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
                j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR'], len(cand[0]['rows'])), flush=True)
    three = '+'.join(map(str, ALL))
    prefixed = {f'{t}_three_seed_label': judgements[f'{t}:C2MS:C0@{three}']['label'] for t in TARGETS} | {f'{t}_seed42_label': judgements[f'{t}:C2MS:C0@42']['label'] for t in TARGETS}
    controller = {}
    for key, m in models.items():
        for target in m['targets']:
            name = 'RR' if target in TRAINED else f'{target}_RR'
            rec = run_dir(a.runtime, *key.split('_')[:2], int(key.split('_s')[-1])) / f'evaluation/{name}.json'
            if a.limit is None and rec.exists():
                ref = json.loads(rec.read_text(encoding='utf-8'))['mean']
                controller[f'{key}:{target}'] = max(abs(m['targets'][target]['mean']['RR'][k] - ref[k]) for k in ('PSNR', 'SAM', 'ERGAS'))
    print('pre-fixed', prefixed, 'checks', checks, 'controller RR max abs diff', max(controller.values(), default=None), flush=True)
    for target, sensor in TARGETS.items():
        s = SENSORS[target]
        result_images.render(nets[f'C0_{sensor}_s42'], nets[f'C2MS_{sensor}_s42'], a.data, out / f'images/{target}_C2MS_vs_original', names=('원본', 'C2MS'), seed=42,
                             device=a.device, scenes=tuple(i for i in CFG['scenes'] if a.limit is None or i < a.limit), labels=('원본 SSA-MRN', '현재 기준 C2MS'),
                             rr_file=files(target)['RR'], scale=s['max_value'], bands=s['rgb'], land_cover={})
    graphs(models, judgements, a.runtime, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3 with sensor-specific Wald operators (prepare.py)', noise=NOISE, tolerances=TOL, seeds=list(ALL), targets=TARGETS,
        scenes_per_target=a.limit or 20, prefixed_judgement=prefixed, judgements=judgements, checks=checks, controller_rr_max_abs_diff=controller, models=models,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=sha(__file__),
                         prepare_sha256=sha(HERE / 'prepare.py'), protocol_sha256=sha(ROOT / 'common/evaluation_protocol.py'),
                         result_images_sha256=sha(ROOT / 'common/result_images.py'), data_sha256=data_sha)),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
