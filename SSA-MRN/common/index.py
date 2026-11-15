"""Rebuild the experiment tables from experiments/catalog.json (the single source of the experiment list).

  python SSA-MRN/common/index.py          rewrite the table in experiments/<track>/README.md
  python SSA-MRN/common/index.py --check  fail if a table is stale or a catalog entry is incomplete

Catalog keys are '<track>/<number>_<name>'. Each entry needs display_name, experimenter, status, improvement and
result (one line, at most 100 characters). Tables are written between the experiments:start / experiments:end markers.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'experiments/catalog.json'
START, END = '<!-- experiments:start -->', '<!-- experiments:end -->'
STATUS = ('계획', '학습 중', '학습 중단', '학습 완료·평가 대기', '학습 완료·일부 평가', '평가 완료', '사전 검증 완료')
IMPROVEMENT = ('개선', '부분 개선', '개선 없음', '판단 보류', '비교 기준', '성능 비교 아님')
REQUIRED = ('display_name', 'experimenter', 'status', 'improvement', 'result')


def problems(catalog):
    found = []
    for key, entry in catalog.items():
        folder = ROOT / 'experiments' / key
        if not (folder / 'README.md').is_file(): found.append(f'{key}: no README.md')
        for field in REQUIRED:
            if not entry.get(field): found.append(f'{key}: missing {field}')
        if entry.get('status') not in STATUS: found.append(f'{key}: status {entry.get("status")!r}')
        if entry.get('improvement') not in IMPROVEMENT: found.append(f'{key}: improvement {entry.get("improvement")!r}')
        if len(entry.get('result', '')) > 100: found.append(f'{key}: result longer than 100 characters')
    listed = set(catalog)
    for folder in (ROOT / 'experiments').glob('*/[0-9]*_*'):
        if f'{folder.parent.name}/{folder.name}' not in listed: found.append(f'{folder.parent.name}/{folder.name}: not in catalog')
    return found


def table(catalog, track):
    rows = sorted(((int(key.split('/')[1].split('_')[0]), key, entry) for key, entry in catalog.items() if key.startswith(track + '/')), reverse=True)
    lines = ['| 실험 | 핵심 결과 | 상태 | 판정 |', '|---|---|:---:|:---:|']
    for _, key, entry in rows:
        verdict = entry['improvement'].replace(' ', '&nbsp;')
        if entry['improvement'] == '개선': verdict = f'**{verdict}**'
        lines.append('| [{}]({}/README.md)<br>{} | {} | {} | {} |'.format(
            entry['display_name'], key.split('/')[1], entry['experimenter'], entry['result'].replace('|', '\\|'),
            entry['status'].replace(' ', '&nbsp;'), verdict))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    found = problems(catalog)
    stale = []
    for track in sorted({key.split('/')[0] for key in catalog}):
        page = ROOT / 'experiments' / track / 'README.md'
        text = page.read_text(encoding='utf-8')
        if START not in text or END not in text: found.append(f'{track}/README.md: no table markers'); continue
        new = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: f'{START}\n{table(catalog, track)}\n{END}', text, flags=re.S)
        if new != text:
            stale.append(track)
            if not args.check: page.write_text(new, encoding='utf-8')
    for line in found: print('ERROR', line)
    if args.check and stale: print('STALE tables:', ', '.join(stale))
    if found or (args.check and stale): sys.exit(1)
    print(f'OK: {len(catalog)} experiments' + (f', rewrote {", ".join(stale)}' if stale else ''))


if __name__ == '__main__': main()
