"""Experiment 53 evaluation (no training): criteria v2.3 judgements, result images, FR/RR2 panels and graphs.

Measures F1 (49 seed 42, current baseline), C2 (43 seed 42), C2MS and F1MS (this experiment, seed 42) with
common/evaluation_protocol.scene_metrics on all 20 RR and FR test scenes, then applies the judgement fixed in config.json.
Also checks the current prepare.py against the preserved coarse.py and the measured RR means against the controller's
own evaluation records. Writes only below results/.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from common import evaluation_protocol as protocol, result_images  # noqa: E402
sys.path.insert(0, str(HERE))
import model as M53  # noqa: E402
import prepare  # noqa: E402

E = ROOT / 'experiments/performance'
CKPT = {'F1': E / '49_observation_refinement_seed_repeat/checkpoints/F1_QB_s42/best.pt', 'C2': E / '43_normalized_parallel_capacity/checkpoints/C2_QB_s42/best.pt',
        'C2MS': HERE / 'checkpoints/C2MS_QB_s42/best.pt', 'F1MS': HERE / 'checkpoints/F1MS_QB_s42/best.pt'}
BUILD = {'F1': 'F1', 'C2': 'C2', 'C2MS': 'C2', 'F1MS': 'F1'}
HISTORY = {'F1': E / '49_observation_refinement_seed_repeat/results/F1_QB_s42/history.json', 'C2': E / '43_normalized_parallel_capacity/results/C2_QB_s42/history.json',
           'C2MS': HERE / 'results/C2MS_QB_s42/history.json', 'F1MS': HERE / 'results/F1MS_QB_s42/history.json'}
NOISE = {1: 0.06, 2: 0.04}  # docs/judgement_criteria.md v2.3: RR PSNR noise threshold for 1 seed / 2 or more seeds
TOL = {'RR2': (-0.05, 0.06), 'FR_WALD': (-0.07, 0.04), 'FR_OBS': 1.10}
PAIRS = ('C2MS:F1', 'C2MS:C2', 'F1MS:F1', 'C2MS:F1MS')
VISUAL_SCENES = (1, 8, 11, 13, 19)  # FR/RR2 panels fixed in config.json before training
COLORS = {'F1': '#4C72B0', 'C2': '#8C8C8C', 'C2MS': '#C44E52', 'F1MS': '#55A868'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def judge(cand, ctrl):
    """v2.3 on seed means (lists of per-seed mean dicts)."""
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
    out = {}
    for part in ('RR', 'RR2', 'FR_WALD'):
        out[part] = {'PSNR': int(sum(c[part]['PSNR'] > b[part]['PSNR'] for c, b in zip(cand_rows, ctrl_rows))),
                     'SAM': int(sum(c[part]['SAM'] < b[part]['SAM'] for c, b in zip(cand_rows, ctrl_rows)))}
    return out


def current_code_check():
    """prepare.py (current) must produce exactly the pairs of the preserved coarse.py that built the training cache."""
    spec = importlib.util.spec_from_file_location('coarse_original', HERE / 'provenance/code/coarse.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    g = torch.Generator().manual_seed(53)
    pan, gt = torch.rand(4, 1, 64, 64, generator=g), torch.rand(4, 4, 64, 64, generator=g)
    a, b = prepare.CoarsePairs('cpu')(pan, gt), original.CoarsePairs('cpu')(pan, gt)
    return {'prepare_vs_preserved_coarse_max_abs': max(float((x - y).abs().max()) for x, y in zip(a, b)), 'shapes': [list(x.shape) for x in a]}


def panels(nets, data, out, device, degrade):
    """FR (centre 128x128 of 512) and RR2 (whole 64x64) panels for F1, C2, C2MS, F1MS on the pre-fixed scenes."""
    out.mkdir(parents=True, exist_ok=True); bands = [2, 1, 0]; stats = {}
    def rgb(x, lo, hi): return Image.fromarray((np.clip((x[bands].transpose(1, 2, 0) - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype('uint8'))
    def heat(m, vmax):
        v = np.clip(m / max(vmax, 1e-8), 0, 1)
        return Image.fromarray((np.stack([np.clip(1.5 * v, 0, 1), np.clip(1.5 * v - .5, 0, 1), np.clip(3 * v - 2, 0, 1)], -1) * 255).astype('uint8'))
    def nearest(ms, size): f = size // ms.shape[-1]; return ms.repeat(f, -2).repeat(f, -1)
    def sheet(cells, title, path, zoom):
        w, h = cells[0][1].size; W, H = w * zoom, h * zoom; cols = len(cells) // 2
        img = Image.new('RGB', (cols * W, 2 * (H + 22) + 24), 'white'); d = ImageDraw.Draw(img); d.text((6, 4), title, fill='black')
        for k, (label, im) in enumerate(cells):
            r, c = divmod(k, cols); x, y = c * W, 24 + r * (H + 22); d.text((x + 6, y + 4), label, fill='black'); img.paste(im.resize((W, H), Image.Resampling.NEAREST), (x, y + 22))
        img.save(path)
    with h5py.File(next(data.rglob('test_qb_OrigScale_multiExm1.h5')), 'r') as fr, h5py.File(next(data.rglob('test_qb_multiExm1.h5')), 'r') as rr:
        for i in VISUAL_SCENES:
            a = {k: np.asarray(fr[k][i], dtype=np.float64) for k in ('pan', 'lms', 'ms')}
            outs = {k: protocol.predict(n, a['pan'], a['lms'], a['ms'], device) for k, n in nets.items()}
            c = slice(192, 320); lo, hi = np.percentile(a['lms'][bands][:, c, c].transpose(1, 2, 0), [1, 99], axis=(0, 1))
            p = a['pan'][0][c, c]; pl, ph = np.percentile(p, [1, 99])
            pan_img = Image.fromarray((np.clip((p - pl) / max(ph - pl, 1e-8), 0, 1) * 255).astype('uint8')).convert('RGB')
            det = {k: np.abs(o - a['lms']).mean(0)[c, c] for k, o in outs.items()}; vmax = max(np.percentile(v, 99) for v in det.values())
            blank = Image.new('RGB', pan_img.size, 'white')
            cells = [('PAN', pan_img), ('MS (nearest)', rgb(nearest(a['ms'], 512)[:, c, c], lo, hi))] + [(k, rgb(o[:, c, c], lo, hi)) for k, o in outs.items()]
            cells += [('', blank), ('|out - LMS| scale %.0f DN' % vmax, blank)] + [(f'{k} injected detail', heat(det[k], vmax)) for k in outs]
            sheet(cells, f'FR scene {i} (no ground truth), centre crop 128x128 of 512x512, seed 42', out / f'FR_scene_{i:02}.png', 3)
            stats[f'FR_{i}'] = {k: {'mean_abs_detail_DN': float(np.abs(o - a['lms']).mean())} for k, o in outs.items()}
            b = {k: np.asarray(rr[k][i], dtype=np.float64) for k in ('pan', 'ms')}
            pan_b, lms_b, ms_b = degrade(b['pan'], b['ms']); gt = b['ms']
            outs2 = {k: protocol.predict(n, pan_b, lms_b, ms_b, device) for k, n in nets.items()}
            lo, hi = np.percentile(gt[bands].transpose(1, 2, 0), [1, 99], axis=(0, 1))
            err = {k: np.abs(o - gt).mean(0) for k, o in outs2.items()}; vmax = max(np.percentile(v, 99) for v in err.values())
            blank = Image.new('RGB', (64, 64), 'white')
            cells = [('GT (RR MS)', rgb(gt, lo, hi)), ('MS input (nearest)', rgb(nearest(ms_b, 64), lo, hi))] + [(k, rgb(o, lo, hi)) for k, o in outs2.items()]
            cells += [('', blank), ('|out - GT| scale %.0f DN' % vmax, blank)] + [(f'{k} error', heat(err[k], vmax)) for k in outs2]
            sheet(cells, f'RR2 scene {i} (GT available), full 64x64, seed 42', out / f'RR2_scene_{i:02}.png', 6)
            stats[f'RR2_{i}'] = {k: {'PSNR': protocol.rr_metrics(gt, o)['PSNR'], 'mean_abs_err_DN': float(err[k].mean())} for k, o in outs2.items()}
    (out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')


def graphs(models, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, path in HISTORY.items():
        h = [r for r in json.loads(path.read_text(encoding='utf-8')) if r.get('val_psnr') is not None]
        ax[0].plot([r['epoch'] for r in h], [r['val_psnr'] for r in h], color=COLORS[k], label=k)
        c = [r for r in h if r.get('coarse_val_psnr') is not None]
        if c: ax[1].plot([r['epoch'] for r in c], [r['coarse_val_psnr'] for r in c], color=COLORS[k], label=k)
        t = [r for r in h if r.get('train_mse')]
        ax[2].semilogy([r['epoch'] for r in t], [r['train_mse'] for r in t], color=COLORS[k], label=k)
    ax[0].set(xlabel='epoch', ylabel='PSNR (dB)', ylim=(37, 41), title='Validation PSNR, original scale (selection)')
    ax[1].set(xlabel='epoch', ylabel='PSNR (dB)', title='Validation PSNR, 2x coarser pairs (logged only)')
    ax[2].set(xlabel='epoch', ylabel='MSE (DN/2047)^2', title='Training MSE, original-scale batch')
    for x in ax: x.legend(fontsize=8); x.grid(alpha=.3)
    fig.suptitle('Seed 42 learning curves (F1 from 49, C2 from 43, C2MS and F1MS from this experiment)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'learning.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); tests = ('RR', 'RR2', 'FR_WALD'); x = np.arange(len(tests)); w = .26
    for j, k in enumerate(('C2', 'C2MS', 'F1MS')):
        v = [models[k]['mean'][t]['PSNR'] - models['F1']['mean'][t]['PSNR'] for t in tests]
        ax.bar(x + (j - 1) * w, v, w, color=COLORS[k], label=k)
        for xi, vi in zip(x + (j - 1) * w, v): ax.text(xi, vi + (0.03 if vi >= 0 else -0.09), f'{vi:+.2f}', ha='center', fontsize=8)
    ax.axhline(0, color='black', lw=.8); ax.set_xticks(x, tests); ax.set_ylabel('PSNR difference vs F1 (dB)'); ax.set_title('Seed 42, mean of 20 test scenes, vs current baseline F1')
    ax.legend(); ax.grid(axis='y', alpha=.3); fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--graphs-only', action='store_true', help='redraw graphs from results/evaluation_v2_3.json')
    a = p.parse_args()
    out = HERE / 'results'
    if a.graphs_only:
        return graphs(json.loads((out / 'evaluation_v2_3.json').read_text(encoding='utf-8'))['models'], out)
    if (out / 'evaluation_v2_3.json').exists(): raise FileExistsError(out / 'evaluation_v2_3.json')
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    scenes = {}
    for name, filename in (('RR', 'test_qb_multiExm1.h5'), ('FR', 'test_qb_OrigScale_multiExm1.h5')):
        found = list(a.data.rglob(filename)); assert len(found) == 1
        with h5py.File(found[0], 'r') as f:
            keys = ('pan', 'lms', 'ms', 'gt') if name == 'RR' else ('pan', 'lms', 'ms')
            scenes[name] = [{k: np.asarray(f[k][i], dtype=np.float64) for k in keys} for i in range(len(f['pan']))]
        scenes[name + '_sha256'] = sha(found[0])
    degrade = protocol.WaldDegradation('QB', a.device)
    observation = protocol.SensorObservation('QB', 'decimate', (2, 2)).to(a.device).double()
    nets, models = {}, {}
    for key, path in CKPT.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = M53.M48.Model(BUILD[key], 42); net.load_state_dict(ck['model'], strict=True); nets[key] = net.to(a.device).eval()
        rows = [{'index': i, **protocol.scene_metrics(nets[key], rr, fr, a.device, degrade, observation)} for i, (rr, fr) in enumerate(zip(scenes['RR'], scenes['FR']))]
        mean = {part: {k: float(np.mean([r[part][k] for r in rows])) for k in rows[0][part]} for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
        models[key] = dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path), best_epoch=ck.get('best_epoch'), mean=mean, rows=rows)
        print(key, 'RR %.4f/%.4f/%.4f RR2 %.4f/%.4f FR_WALD %.4f/%.4f OBS %.3f' % (mean['RR']['PSNR'], mean['RR']['SAM'], mean['RR']['ERGAS'], mean['RR2']['PSNR'],
              mean['RR2']['SAM'], mean['FR_WALD']['PSNR'], mean['FR_WALD']['SAM'], mean['FR']['OBS_RMSE_DN']), flush=True)
    judgements = {}
    for pair in PAIRS:
        cand, ctrl = pair.split(':')
        j = judge([models[cand]['mean']], [models[ctrl]['mean']]); j['scenes_improved'] = scene_wins(models[cand]['rows'], models[ctrl]['rows']); judgements[pair] = j
        print(pair, j['label'], 'RR %+.4f/%+.4f/%+.4f RR2 %+.4f/%+.4f FR_WALD %+.4f/%+.4f OBS x%.3f RR scenes %d/20' % (
            j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR']['ERGAS'], j['delta']['RR2']['PSNR'], j['delta']['RR2']['SAM'],
            j['delta']['FR_WALD']['PSNR'], j['delta']['FR_WALD']['SAM'], j['obs_ratio'], j['scenes_improved']['RR']['PSNR']), flush=True)
    primary = judgements['C2MS:F1']
    prefixed = {'primary_label': primary['label'],
                'large_gain_condition': primary['label'] == '개선' and primary['delta']['RR']['PSNR'] >= 0.40,
                'diagnostic_1': judgements['C2MS:C2']['delta']['RR']['PSNR'] >= -0.15 and primary['guards']['RR2'],
                'diagnostic_2_label': judgements['F1MS:F1']['label']}
    controller = {k: max(abs(models[k]['mean']['RR'][m] - json.loads((out / f'{k}_QB_s42/evaluation/RR.json').read_text(encoding='utf-8'))['mean'][m]) for m in ('PSNR', 'SAM', 'ERGAS')) for k in ('C2MS', 'F1MS')}
    code = current_code_check()
    print('pre-fixed', prefixed, 'controller RR max abs diff', controller, 'current code', code, flush=True)
    for cand in ('C2MS', 'F1MS'):
        result_images.render(nets['F1'], nets[cand], a.data, out / f'images/{cand}_vs_F1', names=('F1', cand), seed=42, device=a.device)
    panels(nets, a.data, out / 'fr_rr2_visual', a.device, degrade)
    graphs(models, out)
    (out / 'evaluation_v2_3.json').write_text(json.dumps(dict(
        criteria='docs/judgement_criteria.md v2.3', noise=NOISE, tolerances=TOL, seeds=[42], prefixed_judgement=prefixed, judgements=judgements,
        controller_rr_max_abs_diff=controller, current_code_check=code, models=models,
        environment=dict(python=sys.version, torch=torch.__version__, device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu', evaluate_sha256=sha(__file__),
                         protocol_sha256=sha(ROOT / 'common/evaluation_protocol.py'), result_images_sha256=sha(ROOT / 'common/result_images.py'),
                         rr_sha256=scenes['RR_sha256'], fr_sha256=scenes['FR_sha256'])),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
