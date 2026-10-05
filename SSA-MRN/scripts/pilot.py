"""List and manually launch validation-only pilot jobs; never auto-run the suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'start', 'status', 'logs', 'resume-latest', 'resume-best'))
    parser.add_argument('label', nargs='?')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--from-dir', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'SSA-MRN/configs/pilot/suite.json').read_text(encoding='utf-8-sig'))
    if args.command == 'list':
        for job in manifest['jobs']:
            print(f"{job['label']:24s} {job['purpose']}")
        return
    command = [sys.executable, str(ROOT / 'scripts/remote-training-pilot.py'), args.command]
    if args.command in ('start', 'resume-latest', 'resume-best'):
        jobs = {job['label']: job for job in manifest['jobs']}
        if args.label not in jobs:
            parser.error('Choose a valid label from pilot.py list')
        command += ['--config', str(ROOT / jobs[args.label]['config'])]
    elif args.label:
        parser.error('A label is used only for start/resume')
    if args.follow:
        if args.command != 'logs':
            parser.error('--follow requires logs')
        command.append('--follow')
    if args.from_dir:
        if not args.command.startswith('resume-'):
            parser.error('--from-dir requires resume')
        command += ['--from-dir', str(args.from_dir)]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == '__main__':
    main()
