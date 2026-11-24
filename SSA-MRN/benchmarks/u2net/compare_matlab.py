"""Compare saved U2Net predictions and SMOC using actual MATLAB only."""
import argparse
import json
from pathlib import Path
import statistics

KEYS = ('PSNR_bandmean_peak2047', 'SAM', 'ERGAS', 'SCC', 'Q2n', 'Q')
SENSORS = ('QB', 'GF2', 'WV3', 'WV2')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cases = {}
    for path in a.input.rglob('summary.json'):
        item = json.loads(path.read_text(encoding='utf-8'))
        if 'matlab_mean' not in item: continue
        case = item['case']; key = (case['model'], case['sensor'])
        if key in cases: raise ValueError('Duplicate MATLAB case: ' + str(key))
        if item['scenes'] != 20: raise ValueError('Incomplete MATLAB case')
        if item['dlpan_commit'] != 'a34f884af88967580e3ef4f61a89e12fdfe4cda5': raise ValueError('Unexpected MATLAB source')
        if item['matlab_release'] != '2025b' and item['matlab_release'] != 'R2025b':
            raise ValueError('Unexpected MATLAB release: ' + str(item['matlab_release']))
        cases[key] = item
    expected = {('u2net', s) for s in SENSORS} | {('smoc', s) for s in SENSORS} | {(m, s) for m in ('smoc_s43', 'smoc_s44') for s in ('WV3', 'WV2')}
    if set(cases) != expected: raise ValueError('Missing/extra MATLAB cases: ' + str(expected.symmetric_difference(cases)))
    result = dict(metric_source='actual MATLAB R2025b / DLPan a34f884 / score_suite.m', comparisons={})
    lines = ['| 센서 | U2Net 설정 | SMOC 비교 시드 | U2Net PSNR | SMOC PSNR | ΔPSNR | U2Net SAM | SMOC SAM | ΔSAM | U2Net ERGAS | SMOC ERGAS | ΔERGAS |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sensor in SENSORS:
        u2 = cases['u2net', sensor]
        names = ['smoc'] if sensor in ('QB', 'GF2') else ['smoc', 'smoc_s43', 'smoc_s44']
        ref = [cases[n, sensor]['matlab_mean'] for n in names]
        mean = {k: statistics.mean(x[k] for x in ref) for k in KEYS}
        std = {k: statistics.stdev(x[k] for x in ref) if len(ref)>1 else None for k in KEYS}
        cand = {k: u2['matlab_mean'][k] for k in KEYS}
        delta = {k: cand[k]-mean[k] for k in KEYS}
        result['comparisons'][sensor] = dict(setting=u2['case']['setting'], candidate_seed=42 if sensor in ('QB','GF2') else None,
            baseline_seeds=[42] if len(ref)==1 else [42,43,44], candidate=cand, smoc_mean=mean, smoc_std=std, difference=delta,
            comparison='paired seed42' if len(ref)==1 else 'unknown author seed vs SMOC three-seed metric mean',
            transfer=sensor=='WV2', python_same_arrays=u2['python_mean'], matlab_minus_python=u2['matlab_minus_python'])
        lines.append(f'| {sensor} | {u2["case"]["setting"]} | {"42" if len(ref)==1 else "42·43·44 평균"} | {cand[KEYS[0]]:.4f} | {mean[KEYS[0]]:.4f} | {delta[KEYS[0]]:+.4f} | {cand["SAM"]:.4f} | {mean["SAM"]:.4f} | {delta["SAM"]:+.4f} | {cand["ERGAS"]:.4f} | {mean["ERGAS"]:.4f} | {delta["ERGAS"]:+.4f} |')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'smoc_comparison_matlab.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (a.out/'matlab_comparison_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__': main()
