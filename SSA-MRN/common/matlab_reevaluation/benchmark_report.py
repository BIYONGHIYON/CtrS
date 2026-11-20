"""Summarize actual-MATLAB scores of a benchmark model's RR predictions next to the CtrS Python metrics of the same arrays.

case:      one sensor folder (prediction_manifest.json, matlab_scores.json, python_metrics.json) -> summary.json
aggregate: all case summaries -> matlab_summary.json and matlab_table.md (also printed to the job log)

PSNR: score_suite.m computes PSNR at the sensor peak (GF2 1023, others 2047), both from the global MSE and as the band mean. The
CtrS Python PSNR (common/metrics) is the band mean at peak 2047 for every sensor, so for GF2 the MATLAB band mean is also given at
peak 2047 (+20 log10(2047/1023) dB) to compare like with like. No paper number is changed or chosen here.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from suite import read, write  # noqa: E402

GF2_TO_2047 = 20 * math.log10(2047 / 1023)


def case(out):
    out = Path(out); meta = read(out / 'prediction_manifest.json'); matlab = read(out / 'matlab_scores.json'); py = read(out / 'python_metrics.json')
    rows = matlab['rows'] if isinstance(matlab['rows'], list) else [matlab['rows']]
    if sorted(r['scene'] for r in rows) != list(range(20)) or any(r['protocol'] != 'RR' for r in rows): raise ValueError('incomplete MATLAB RR scenes')
    m = {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
    p = {k: float(np.mean([r['python'][k] for r in py])) for k in py[0]['python']}
    sensor = meta['case']['sensor']
    m['PSNR_bandmean_peak2047'] = m['PSNR_bandmean'] + (GF2_TO_2047 if sensor == 'GF2' else 0.0)
    diff = {k: m[k] - p[k] for k in ('SAM', 'ERGAS', 'SCC', 'Q2n')}
    diff['PSNR'] = m['PSNR_bandmean_peak2047'] - p['PSNR']
    result = dict(case=meta['case'], scenes=len(rows), matlab_mean=m, python_mean=p, matlab_minus_python=diff, peak=meta['peak'],
                  matlab_release=matlab.get('matlab_release'), dlpan_commit=matlab.get('dlpan_commit'), profile=matlab.get('profile'),
                  data=meta['data'], prediction_files=meta['files'],
                  note='MATLAB Q is the DLPan band-averaged Q; Q2n is Q4 (4 bands) or Q8 (8 bands), the Q index papers report; '
                       'with bands=rgb the 3 RGB bands are padded with one zero band to Q4 (DLPan q2n)')
    write(out / 'summary.json', result)
    print(json.dumps(dict(case=meta['case']['id'], matlab=m, python=p, matlab_minus_python=diff), indent=1), flush=True)


def aggregate(folder, out):
    rows = []
    for p in sorted(Path(folder).rglob('summary.json')):
        s = read(p)
        if 'matlab_mean' in s: rows.append(s)
    order = {'QB': 0, 'GF2': 1, 'WV3': 2, 'WV2': 3}
    rows.sort(key=lambda s: order.get(s['case']['sensor'], 9))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    write(out / 'matlab_summary.json', dict(cases=rows))
    lines = ['| 센서 | MATLAB PSNR (밴드 평균, peak 2047) | MATLAB SAM | MATLAB ERGAS | MATLAB Q2n | MATLAB Q | Python PSNR | Python SAM | Python ERGAS | MATLAB−Python (PSNR / SAM / ERGAS / Q2n) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in rows:
        m, p, d = s['matlab_mean'], s['python_mean'], s['matlab_minus_python']
        lines.append(f"| {s['case']['sensor']} | {m['PSNR_bandmean_peak2047']:.3f} | {m['SAM']:.4f} | {m['ERGAS']:.4f} | {m['Q2n']:.4f} | {m['Q']:.4f} | "
                     f"{p['PSNR']:.3f} | {p['SAM']:.4f} | {p['ERGAS']:.4f} | {d['PSNR']:+.4f} / {d['SAM']:+.4f} / {d['ERGAS']:+.4f} / {d['Q2n']:+.4f} |")
    (out / 'matlab_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    for s in rows:
        print(s['case']['id'], 'MATLAB', json.dumps({k: round(v, 5) for k, v in s['matlab_mean'].items()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('case'); c.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('aggregate'); g.add_argument('--input', type=Path, required=True); g.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    case(a.out) if a.command == 'case' else aggregate(a.input, a.out)


if __name__ == '__main__':
    main()
