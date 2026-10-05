"""CPU unit checks and weight-only initialization audit; never starts training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def call(arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'training_started': False, 'checks': {}}
    for name in ('test_pilot_suite.py', 'test_joint17_consistency.py', 'test_area_consistency.py'):
        out = call([str(ROOT / 'SSA-MRN/tests' / name)])
        result['checks'][name] = out.stderr.strip()
        print(name, 'PASS', flush=True)
    result['environment'] = json.loads(call([str(ROOT / 'SSA-MRN/scripts/check_pilot_environment.py')]).stdout)
    for label in ('02_warm_control10', '02_warm_low10'):
        out = call([str(ROOT / 'SSA-MRN/scripts/train_lib.py'), '--config',
                    str(ROOT / 'SSA-MRN/configs/pilot' / (label + '.json')),
                    '--device', 'cpu', '--check-only', '--output-dir',
                    str(ROOT / 'SSA-MRN/experiments/checkpoints/setup-checks' / label)])
        check = json.loads(out.stdout)
        if check['next_epoch'] != 1 or check['optimizer_state_entries'] != 0:
            raise RuntimeError('Weight-only initialization did not reset optimizer/epoch')
        if check['initialization']['sha256'] != result['environment']['checkpoint_sha256']:
            raise RuntimeError('Initialization checkpoint mismatch')
        result['checks'][label] = check
        print(label, 'weights-only PASS', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Training was NOT started.', flush=True)


if __name__ == '__main__':
    main()
