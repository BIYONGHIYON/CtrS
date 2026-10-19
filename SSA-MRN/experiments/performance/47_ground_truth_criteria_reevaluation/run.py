"""Experiment 47 driver: evaluate every group in its own process, then apply criteria v2."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
DIRECTION = {'PSNR': 1, 'SAM': -1, 'ERGAS': -1}


def seed_mean(group, arm, part, metric):
    return float(np.mean([group['models'][f'{arm}_s{s}']['mean'][part][metric] for s in group['seeds']]))


def judge(group, candidate):
    control, seeds = group['spec']['control'], group['seeds']
    d = {part: {k: seed_mean(group, candidate, part, k) - seed_mean(group, control, part, k) for k in group['models'][f'{control}_s{seeds[0]}']['mean'][part]}
         for part in ('RR', 'RR2', 'FR_WALD', 'FR')}
    rr_ok = [DIRECTION[k] * d['RR'][k] > 0 for k in ('PSNR', 'SAM', 'ERGAS')]
    psnr_seeds = sum(group['models'][f'{candidate}_s{s}']['mean']['RR']['PSNR'] > group['models'][f'{control}_s{s}']['mean']['RR']['PSNR'] for s in seeds)
    primary = all(rr_ok) and (len(seeds) < 3 or psnr_seeds >= 2)
    obs_ratio = seed_mean(group, candidate, 'FR', 'OBS_RMSE_DN') / seed_mean(group, control, 'FR', 'OBS_RMSE_DN')
    guards = {'RR2': d['RR2']['PSNR'] >= -0.05 and d['RR2']['SAM'] <= 0.02,
              'FR_WALD': d['FR_WALD']['PSNR'] >= -0.05 and d['FR_WALD']['SAM'] <= 0.02,
              'FR_OBS': obs_ratio <= 1.10}
    if primary and all(guards.values()): label = '개선'
    elif primary or any(rr_ok): label = '부분 개선'
    else: label = '개선 없음'
    return {'delta': d, 'obs_ratio': obs_ratio, 'rr_components_improved': dict(zip(('PSNR', 'SAM', 'ERGAS'), rr_ok)), 'psnr_seeds_improved': int(psnr_seeds),
            'primary': primary, 'guards': guards, 'label': label, 'seeds': seeds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='needed only when a group result is missing')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--groups', nargs='*')
    parser.add_argument('--tiers', type=int, nargs='*', default=[1, 2, 3])
    a = parser.parse_args()
    a.groups = a.groups or [g for g, spec in CONFIG['groups'].items() if spec.get('tier', 1) in a.tiers]
    a.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for g in a.groups:
        if (a.output / f'{g}.json').exists():
            continue
        if a.data is None: raise SystemExit(f'group {g} needs --data')
        code = subprocess.run([sys.executable, '-u', str(HERE / 'evaluate.py'), '--group', g, '--data', str(a.data), '--output', str(a.output), '--device', a.device]).returncode
        if code: failed.append(g); print(f'group {g} failed with exit code {code}', flush=True)
    summary = {}
    for g in CONFIG['groups']:
        path = a.output / f'{g}.json'
        if not path.exists(): continue
        group = json.loads(path.read_text(encoding='utf-8'))
        summary[g] = {c: judge(group, c) for c in group['spec']['candidates']}
    (a.output / 'summary.json').write_text(json.dumps({'criteria': CONFIG['criteria_v2'], 'judgements': summary, 'time': time.time()}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for g, cands in summary.items():
        for c, j in cands.items():
            print(g, c, j['label'], 'RR dPSNR %+.4f dSAM %+.4f | RR2 dPSNR %+.4f | FR_WALD dPSNR %+.4f | OBS x%.3f | QNR %+.4f' % (
                j['delta']['RR']['PSNR'], j['delta']['RR']['SAM'], j['delta']['RR2']['PSNR'], j['delta']['FR_WALD']['PSNR'], j['obs_ratio'], j['delta']['FR']['QNR']), flush=True)
    if failed: raise SystemExit(f'failed groups: {failed}')


if __name__ == '__main__':
    main()
