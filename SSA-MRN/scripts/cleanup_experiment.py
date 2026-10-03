"""Prune an explicitly completed run, preserving weights, metrics and compact epoch records."""
import argparse
import json
from pathlib import Path
from report_graphs import compact_history


def cleanup_plan(run):
    run = run.resolve()
    if not run.is_dir():
        raise ValueError('Run directory is missing')
    if not (run/'latest.pt').is_file() and not (run/'best.pt').is_file():
        raise ValueError('Expected a trained run containing best.pt or latest.pt')
    # Exclude all other run folders: only direct temporary config/log files qualify.
    return [p for p in run.iterdir() if p.is_file() and
            (p.name in ('run_config.json', 'history.jsonl', 'config.json') or p.suffix == '.log')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--completed', action='store_true', help='Explicitly assert this run is stopped/finished')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    targets = cleanup_plan(args.run_dir)
    if args.apply and not args.completed:
        parser.error('Confirm stopped/finished with --completed; active runs must not be pruned')
    for path in targets:
        print(('DELETE ' if args.apply else 'WOULD DELETE ') + str(path))
    if args.apply:
        history = args.run_dir/'history.jsonl'
        if history.exists():
            summary = compact_history(history)
            if not summary['epochs']:
                raise ValueError('Refusing to remove history without measured epoch records')
            (args.run_dir/'curves.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        for path in targets:
            path.unlink()


if __name__ == '__main__':
    main()
