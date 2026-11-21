"""89 controller continuation without the validation PSNR margin (user decision 2026-11-19 14h: "remove the threshold and measure
everything"). The sealed run.py is imported unchanged; only its sanity() is replaced so that a run is stopped for a nonfinite validation
or a best epoch of 0, but no longer for a validation PSNR more than sanity_margin_db below C2MS-SP. Training code, config, fingerprint
and source hashes are unchanged, so run.run(resume=True) passes its sealed checks.

Started by watch_margin_stop.ps1 only after the original controller exited with code 3 and stopped.json names nothing but the PSNR
margin; otherwise it exits with code 4 without running anything."""
import json,math,os,sys
from pathlib import Path
D89=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D89));sys.path.insert(1,str(D89.parents[2]))
os.chdir(D89)
import run as R

MARGIN='dB below C2MS-SP'


def sanity(cfg,root,key):
    v=R.validation_of(root,key);problems=[]
    if not all(math.isfinite(x) for x in (v['psnr'],v['sam'],v['mse'])):problems.append('nonfinite validation')
    if v['epoch']<1:problems.append('best epoch is 0 (no improvement over the initial state)')
    return problems


def main():
    conf=R.load_conf();root=Path(conf['runtime']);stopped=root/'stopped.json'
    if stopped.exists():
        rec=json.loads(stopped.read_text(encoding='utf8'));reason=rec.get('reason') or ''
        parts=[p for p in reason.split(': ',1)[-1].split('; ') if p]
        if not parts or not all(MARGIN in p for p in parts):
            print('STOP REASON IS NOT ONLY THE PSNR MARGIN; NOT CONTINUING:',reason,flush=True);sys.exit(4)
        k=1
        while (root/f'stopped_margin_{k}.json').exists():k+=1
        os.replace(stopped,root/f'stopped_margin_{k}.json')
    R.write(root/'controller_amendment.json',dict(time_kst=R.now(),decision='user 2026-11-19 14h: remove the validation PSNR margin, measure every candidate',
            removed='validation PSNR more than sanity_margin_db below the same-seed C2MS-SP',kept='process failure, incomplete run, nonfinite validation, best epoch 0, W2 check',
            script=str(Path(__file__).resolve()),script_sha256=R.sha(Path(__file__).resolve())))
    R.sanity=sanity
    R.run(True)


if __name__=='__main__':main()
