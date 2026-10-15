"""Run Phase 0 (prepare.py) then Phase 1 in two parallel slots: (W0, W1) then (W1c, W2).

W2 runs only if the Phase 0 gate passes, or with --force-w2. In the recorded run the
gate failed and W2 was started by the user-requested confirmation (screen/add_w2.py in
provenance/training_source.zip), one minute after W1c.
"""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
RT = sys.argv[1]
FORCE_W2 = '--force-w2' in sys.argv[2:]
A0 = r'C:\CtrS-ssai4\runtime\qb_s42_100\A0\best.pt'
CACHE = r'C:\CtrS-ssai4\runtime\qb_s42_100\cache'
DIAG = os.path.join(RT, 'diag.json')
PY = sys.executable


def log(msg):
    with open(os.path.join(RT, 'queue.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def run_pair(arms):
    procs = []
    for arm in arms:
        f = open(os.path.join(RT, arm + '.log'), 'w')
        procs.append((arm, subprocess.Popen([PY, os.path.join(HERE, 'train.py'), '--arm', arm, '--a0', A0, '--cache', CACHE, '--runtime', RT, '--diag', DIAG], stdout=f, stderr=subprocess.STDOUT), f))
        log('started ' + arm)
        time.sleep(60)  # let the first worker finish its GPU cache upload
    for arm, p, f in procs:
        p.wait(); f.close(); log(f'{arm} exit {p.returncode}')


os.makedirs(RT, exist_ok=True)
if os.path.exists(os.path.join(RT, 'stop.request')): sys.exit(0)
if not os.path.exists(DIAG):
    log('phase0 start')
    with open(os.path.join(RT, 'diag.log'), 'w') as f:
        r = subprocess.run([PY, os.path.join(HERE, 'prepare.py'), '--a0', A0, '--cache', CACHE, '--out', DIAG], stdout=f, stderr=subprocess.STDOUT)
    log(f'phase0 exit {r.returncode}')
    if r.returncode: sys.exit(1)
gate = json.load(open(DIAG))['w2_gate_passed']
log(f'w2 gate {gate}')
run_pair(['W0', 'W1'])
if os.path.exists(os.path.join(RT, 'stop.request')): sys.exit(0)
run_pair(['W1c', 'W2'] if gate or FORCE_W2 else ['W1c'])
log('queue complete')
