"""Experiment 12 dispatcher on the preserved SSAI suite and the unchanged experiment 8 training function.

Order: prepare.py (no training) -> benchmark (discarded probes) -> start. benchmark and start
require explicit user approval; nothing here restarts runs or changes conditions automatically.
"""
import importlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import ssai_suite as suite

PACKAGE = 'experiments.ssai.12_observation_matched_pan_guide'
ARMS = ('B2', 'B3')
CONSTANTS = {}


def load_constants(runtime):
    plan = json.loads((Path(runtime) / 'plan.json').read_text())
    stats = plan['observation_statistics']
    CONSTANTS.update(eps=float(stats['eps']), gain_max=float(stats['gain_max']))


def model(arm, channels=4):
    if arm not in ARMS:
        raise ValueError('Experiment 12 trains B2/B3 only; A0 and N1 are reused')
    if not CONSTANTS:
        raise RuntimeError('Frozen train statistics not loaded')
    cls = importlib.import_module(PACKAGE + '.model').ObservationMatchedNet
    return cls(channels, 4, local=arm == 'B2', **CONSTANTS)


def benchmark(a):
    """Two discarded probes together (40 batches each, weights not retained) at the N1 cache fraction."""
    p = suite.plan(a)
    folder = a.runtime / 'benchmark' / 'gpu_partial32_B2_B3'
    folder.mkdir(parents=True, exist_ok=False)
    processes = []
    for arm in p['arms']:
        log = (folder / (arm + '.log')).open('w')
        cmd = [sys.executable, __file__, 'probe', '--runtime', str(a.runtime), '--arm', arm, '--slots', '2', '--micro', '32', '--cache-mode', 'gpu_partial', '--trial', folder.name]
        processes.append((arm, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = time.monotonic() + 180
    while not all((folder / f'2_{arm}.ready').exists() for arm, _, _ in processes):
        if any(proc.poll() is not None for _, proc, _ in processes) or time.monotonic() > deadline:
            break
        time.sleep(.1)
    (folder / '2.go').touch()
    minimum = suite.gpu_free_mib()
    while any(proc.poll() is None for _, proc, _ in processes):
        minimum = min(minimum, suite.gpu_free_mib()); time.sleep(.5)
    codes = []
    for arm, proc, log in processes:
        codes.append(proc.wait()); log.close()
    if any(codes):
        raise RuntimeError(f'Probe failed: {codes}; training not started')
    probes = [json.loads((folder / f'2_{arm}.json').read_text()) for arm in p['arms']]
    result = {'selected_slots': 2, 'selected_micro_batch': 32, 'selected_cache_mode': 'gpu_partial', 'worker_budget_mib': max(x['peak_reserved_mib'] for x in probes) + 192, 'minimum_headroom_mib': 512, 'observed_minimum_free_mib': minimum, 'aggregate_patches_per_second': sum(x['patches_per_second'] for x in probes), 'probes': probes, 'probe_only_no_retained_weights': True, 'safe': minimum >= 512, 'automatic_batch_or_cache_change': False}
    suite.write(a.runtime / 'benchmark.json', result)
    if not result['safe']:
        raise RuntimeError('Less than 512MiB observed headroom; training not started')


original_start = suite.start


def start(a):
    preflight = json.loads((a.runtime / 'preflight.json').read_text())
    bench = json.loads((a.runtime / 'benchmark.json').read_text())
    if not preflight['passed'] or not bench['safe'] or not (a.runtime / 'diagnostics.json').is_file():
        raise RuntimeError('Preflight, diagnostics and a safe benchmark are required')
    original_start(a)


suite.ARMS = ARMS
suite.model = model
# Identical training/validation/checkpoint function to experiment 8 N1 (the B1 control).
suite.train = importlib.import_module('experiments.ssai.8_spatial_mean_normalization.train').train
suite.benchmark = benchmark
suite.start = start
# Child processes must enter this dispatcher; the shared suite file stays byte-identical.
suite.__file__ = __file__
if __name__ == '__main__':
    if '--runtime' in sys.argv:
        runtime = Path(sys.argv[sys.argv.index('--runtime') + 1])
        if (runtime / 'plan.json').exists():
            load_constants(runtime)
    suite.main()
