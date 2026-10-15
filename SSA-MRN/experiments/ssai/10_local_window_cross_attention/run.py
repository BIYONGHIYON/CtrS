"""Dispatcher on the preserved SSAI suite: cache-fraction benchmark, 2-slot queue, evaluation after training."""
import importlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import ssai_suite as suite

PACKAGE = 'experiments.ssai.10_local_window_cross_attention'
FRACTIONS = (0.85, 0.6, 0.4, 0.2, 0.0)
HEADROOM_MIB = 512
TOTAL_MIB = 8192


def model(arm, channels=4):
    if arm in ('L1', 'L2', 'C'):
        return importlib.import_module(PACKAGE + '.model').LocalWindowSSAINet(arm, channels, 4)
    raise ValueError('Experiment 10 trains L1/L2/C only')


original_make_loader = suite.make_loader


def make_loader(ds, cache_mode, batch_size, generator, device):
    # Only the benchmark probes come through here; training builds its own loader.
    if cache_mode == 'gpu_partial' and 'CTRS_CACHE_FRACTION' in os.environ:
        cls = importlib.import_module(PACKAGE + '.train').FractionGPUCachedLoader
        return cls(ds, batch_size, generator, device, float(os.environ['CTRS_CACHE_FRACTION']))
    return original_make_loader(ds, cache_mode, batch_size, generator, device)


def gpu_used_mib():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    return float(r.stdout.strip().splitlines()[0])


def benchmark(a):
    """Probe the heaviest concurrent pair together at decreasing GPU cache fractions.

    micro/effective batch 32, FP32 and the sample order are fixed; only the GPU-
    resident share of the identical data cache changes. The highest fraction that
    keeps >=512 MiB headroom (by nvidia-smi minimum free and by summed peak
    reserved memory) is selected; GPU cache is preferred as in experiments 4-10.
    """
    # L1 joins while L2 is still running, and both carry the attention buffers,
    # so the heaviest pair L2+L1 (not the first-launched L2+C) sets the cache.
    p = suite.plan(a); pair = ['L2', 'L1']
    root = a.runtime / 'benchmark'; root.mkdir(parents=True, exist_ok=False)
    rows = []; chosen = None
    baseline_used = gpu_used_mib()
    for fraction in FRACTIONS:
        trial = f'f{int(fraction * 100):02d}_' + '_'.join(pair); folder = root / trial; folder.mkdir()
        env = dict(os.environ, CTRS_CACHE_FRACTION=str(fraction)); active = []
        for arm in pair:
            log = (folder / (arm + '.log')).open('w')
            cmd = [sys.executable, __file__, 'probe', '--runtime', str(a.runtime), '--arm', arm, '--slots', '2', '--micro', '32', '--cache-mode', 'gpu_partial', '--trial', trial]
            active.append((arm, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        deadline = time.monotonic() + 300
        while not all((folder / f'2_{arm}.ready').exists() for arm, _, _ in active):
            if any(proc.poll() is not None for _, proc, _ in active) or time.monotonic() > deadline: break
            time.sleep(.1)
        (folder / '2.go').touch(); minimum = suite.gpu_free_mib()
        while any(proc.poll() is None for _, proc, _ in active):
            minimum = min(minimum, suite.gpu_free_mib()); time.sleep(.5)
        codes = []
        for arm, proc, log in active: codes.append(proc.wait()); log.close()
        probes = [json.loads((folder / f'2_{arm}.json').read_text()) for arm in pair] if not any(codes) else []
        summed = baseline_used + sum(x['peak_reserved_mib'] for x in probes) if probes else None
        # The controller launches a second worker only when free >= peak + 192 + 512 MiB.
        safe = not any(codes) and minimum >= HEADROOM_MIB and summed is not None and TOTAL_MIB - summed >= HEADROOM_MIB + 192
        row = {'trial': trial, 'fraction': fraction, 'pair': pair, 'exitcodes': codes, 'observed_minimum_free_mib': minimum, 'baseline_used_mib': baseline_used, 'summed_peak_reserved_plus_baseline_mib': summed, 'probes': probes, 'aggregate_patches_per_second': sum(x['patches_per_second'] for x in probes), 'safe': safe}
        rows.append(row); print(json.dumps(row), flush=True)
        if safe:
            chosen = row; break
    if chosen is None:
        suite.write(a.runtime / 'benchmark.json', {'rows': rows, 'safe': False})
        raise RuntimeError('No cache fraction kept 512 MiB headroom with two workers; training not started')
    budget = max(x['peak_reserved_mib'] for x in chosen['probes']) + 192
    result = {'rows': rows, 'selected_slots': 2, 'selected_micro_batch': 32, 'selected_cache_mode': 'gpu_partial', 'selected_cache_fraction': chosen['fraction'], 'worker_budget_mib': budget, 'minimum_headroom_mib': HEADROOM_MIB, 'aggregate_patches_per_second': chosen['aggregate_patches_per_second'], 'probe_only_no_retained_weights': True, 'safe': True, 'automatic_batch_or_cache_change': False, 'rule': 'highest GPU cache fraction whose two-worker probe passes training, validation and checkpoint reload with >=512 MiB headroom'}
    suite.write(a.runtime / 'benchmark.json', result)
    p['cache_fraction'] = chosen['fraction']; p['optimization_benchmark'] = 'benchmark.json'
    p['vram_safety'] = {'worker_budget_mib': budget, 'minimum_headroom_mib': HEADROOM_MIB, 'wait_before_launch': True, 'automatic_batch_change': False}
    suite.write(a.runtime / 'plan.json', p); suite.write(a.runtime / 'provenance/plan.json', p)


original_controller = suite.controller


def controller(a):
    """Original 2-slot queue (L2, C, then L1 when a slot frees), then full evaluation."""
    original_controller(a)
    if not (a.runtime / 'training_complete.json').exists() or (a.runtime / 'stop.request').exists():
        return
    p = suite.plan(a); output = a.runtime / 'evaluation_final'
    if output.exists(): return
    arms = ['A0', 'A0Z'] + p['arms']
    cmd = [sys.executable, str(Path(__file__).parent / 'evaluate.py'), '--source-root', str(ROOT), '--runtime', str(a.runtime), '--data', p['data_root'], '--output', str(output), '--arms', *arms]
    suite.write(a.runtime / 'evaluation_status.json', {'state': 'running', 'arms': arms, 'started': time.time()})
    with (a.runtime / 'evaluation.log').open('a', encoding='utf-8') as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    suite.write(a.runtime / 'evaluation_status.json', {'state': 'complete' if code == 0 else 'failed', 'exitcode': code, 'arms': arms, 'finished': time.time()})


original_start = suite.start


def start(a):
    preflight = json.loads((a.runtime / 'preflight.json').read_text())
    bench = json.loads((a.runtime / 'benchmark.json').read_text())
    if not preflight['passed'] or not bench['safe']:
        raise RuntimeError('Preflight and a safe benchmark are required')
    original_start(a)


suite.ARMS = ('L1', 'L2', 'C')
suite.model = model
suite.make_loader = make_loader
suite.train = importlib.import_module(PACKAGE + '.train').train
suite.benchmark = benchmark
suite.controller = controller
suite.start = start
# Child processes must enter this dispatcher; the shared suite file stays byte-identical.
suite.__file__ = __file__
if __name__ == '__main__': suite.main()
