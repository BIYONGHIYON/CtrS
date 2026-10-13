"""Read-only one-line viewer. Does not import or modify the training controller."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):
    try:return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError,ValueError,PermissionError):return {}


def read_updates(path, offset):
    if not path.exists():return offset,[]
    with path.open("rb") as f:
        size=path.stat().st_size
        if offset is None or offset>size:
            offset=max(0,size-65536);f.seek(offset)
            if offset:f.readline()
        else:f.seek(offset)
        rows=[]
        while True:
            start=f.tell();line=f.readline()
            if not line or not line.endswith(b"\n"):
                f.seek(start);break
            try:rows.append(json.loads(line))
            except (ValueError,UnicodeDecodeError):pass
        return f.tell(),rows


def format_progress(state,config,last):
    name=state.get("current","waiting")
    epoch=last.get("epoch",0);epochs=config.get("epochs",100)
    batch=last.get("batch",0);batches=last.get("batches",0)
    if "val_mse" in last:
        progress="epoch complete";score=f"val MSE {last['val_mse']:.6f}"
    else:
        percent=100*batch/batches if batches else 0
        progress=f"batch {batch}/{batches} {percent:5.1f}%"
        score=f"MSE {last['train_mse']:.6f}" if "train_mse" in last else "waiting for log"
    status=state.get("status","not started")
    if status=="running" and time.time()-state.get("heartbeat",0)>30:status="stale heartbeat"
    return f"{name} | epoch {epoch}/{epochs} | {progress} | {score} | {status}"


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,default=ROOT/"experiments/controlled_suite_v2");p.add_argument("--once",action="store_true");a=p.parse_args()
    current=None;offset=None;last={}
    try:
        while True:
            state=read_json(a.run_dir/"state.json");name=state.get("current")
            if name!=current:current=name;offset=None;last={}
            directory=Path(state.get("external_directory",str(a.run_dir/name))) if name else None
            config=read_json(directory/"config.json") if name else {}
            if name:
                offset,rows=read_updates(directory/"train.log",offset)
                for row in rows:
                    if "epoch" in row:last=row
            text=format_progress(state,config,last)
            width=max(20,shutil.get_terminal_size((120,24)).columns-1)
            sys.stdout.write("\r"+text[:width].ljust(width));sys.stdout.flush()
            if a.once or state.get("status") in ("finished","failed","stopped_by_user","finished_with_blocked_steps"):
                if state.get("error"):print("\n"+state["error"])
                break
            time.sleep(1)
    except KeyboardInterrupt:pass
    finally:print()

if __name__=="__main__":main()
