"""Preflight then fresh training; refuse existing runtime unless explicit resume."""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
if a.runtime.exists() and not a.resume:raise FileExistsError('Fresh runtime required')
if not a.resume:
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('prepare.py')),'--output',str(a.runtime/'preflight.json')],check=True)
cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--data',str(a.data),'--runtime',str(a.runtime)]
if a.resume:cmd.append('--resume')
subprocess.run(cmd,check=True)
