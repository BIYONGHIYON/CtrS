"""Detached sequential experiment controller. Never reads test data."""
import argparse, json, os, subprocess, sys, time, statistics, hashlib, shutil, base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"experiments"/"controlled_suite_v2"

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); t=path.with_suffix(".tmp"); t.write_text(json.dumps(value,indent=2)); os.replace(t,path)

def alive(pid):
    if not pid: return False
    if os.name=="nt":
        import ctypes
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
        if not h:return False
        ctypes.windll.kernel32.CloseHandle(h);return True
    try: os.kill(pid,0);return True
    except OSError:return False

def source_hashes():
    files=list((ROOT/"src").rglob("*.py"))+[ROOT/"scripts/train.py",ROOT/"scripts/train_controlled.py",Path(__file__).resolve(),ROOT/"references/upstream/network.py"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def run(c, name, resume):
    if source_hashes()!=json.loads((RUN/"source_hashes.json").read_text()): raise RuntimeError("source changed during suite")
    directory=RUN/name; directory.mkdir(parents=True,exist_ok=True)
    if (directory/"complete.json").exists():
        done=json.loads((directory/"complete.json").read_text())
        if done["config"]!=c:raise RuntimeError("completed config mismatch")
        return done["best_mse"]
    config=directory/"config.json"
    if config.exists() and json.loads(config.read_text())!=c:raise RuntimeError("existing config mismatch")
    write(config,c)
    cmd=[sys.executable,str(ROOT/"scripts/train_controlled.py"),"--config",str(config),"--output",str(directory)]
    if (directory/(resume+".pt")).exists():cmd += ["--resume",resume]
    with (directory/"train.log").open("a") as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            write(RUN/"state.json",dict(pid=os.getpid(),training_pid=child.pid,current=name,status="running",heartbeat=time.time()))
            time.sleep(5)
        if child.returncode:raise RuntimeError(f"{name} failed; see train.log")
    return json.loads((directory/"complete.json").read_text())["best_mse"]

def worker(plan,resume):
    wins={4:0,6:0}; scores={}
    for sensor,paths in plan["data"].items():
        for k in [4,6]:
            values=[]
            for seed in plan["seeds"]:
                c=dict(plan["training"],sensor=sensor,train=paths["train"],val=paths["val"],seed=seed,k=k,loss="baseline",weight=0.)
                values.append(run(c,f"01_{sensor}_k{k}_s{seed}",resume))
            scores[f"{sensor}_k{k}"]=statistics.mean(values)
        winner=4 if scores[f"{sensor}_k4"]<=scores[f"{sensor}_k6"] else 6;wins[winner]+=1
    k=4 if wins[4]>=wins[6] else 6
    write(RUN/"baseline_selection.json",dict(k=k,scores=scores,rule="per-sensor mean best validation MSE; majority vote; ties K4"))
    for step,kind in [(2,"spectral"),(3,"consistency"),(4,"edge")]:
        for weight in plan["weights"][kind]:
            for seed in plan["seeds"]:
                paths=plan["data"]["QB"]
                c=dict(plan["training"],sensor="QB",train=paths["train"],val=paths["val"],seed=seed,k=k,loss=kind,weight=weight)
                if kind=="consistency":
                    c["observation"]=plan["observations"]["QB"]
                run(c,f"{step:02}_{kind}_w{weight}_s{seed}",resume)
    write(RUN/"state.json",dict(pid=os.getpid(),status="finished",heartbeat=time.time()))

def validate_plan(plan):
    import torch, h5py
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    if shutil.disk_usage(RUN).free < 20*1024**3:raise RuntimeError("at least 20 GB free disk required")
    for sensor,paths in plan["data"].items():
        if Path(paths["train"]).resolve()==Path(paths["val"]).resolve():raise RuntimeError("train/val overlap")
        for path in paths.values():
            with h5py.File(path,"r") as data:
                if not all(key in data for key in ["pan","lms","ms","gt"]):raise RuntimeError("H5 missing input")
                if data["gt"].shape[1]!=(8 if sensor=="WV3" else 4):raise RuntimeError("sensor/channel mismatch")
    sys.path.insert(0,str(ROOT/"src"))
    from ssamrn.observation import load_profile
    observation=plan.get("observations",{}).get("QB")
    if not observation or not observation.get("sha256"):
        raise RuntimeError("Verified QB observation profile required before the 1-4 suite can start")
    profile_path=Path(observation["profile"])
    if not profile_path.is_absolute():profile_path=ROOT/profile_path
    load_profile(profile_path,observation["sha256"],"QB",plan["data"]["QB"]["train"])
    return {"status":"ready", "steps":[1,2,3,4], "sensors":list(plan["data"]), "observation_sensor":"QB", "optimizer_steps":0}

def main():
    global RUN
    p=argparse.ArgumentParser();p.add_argument("command",choices=["start","resume-latest","resume-best","status","inspect","logs","worker","check"]);p.add_argument("--plan",type=Path,default=ROOT/"scripts/controlled_suite_plan.json");p.add_argument("--follow",action="store_true");p.add_argument("--resume",default="latest",choices=["latest","best"])
    p.add_argument("--run-dir",type=Path,default=RUN)
    a=p.parse_args(); RUN=a.run_dir.resolve();RUN.mkdir(parents=True,exist_ok=True);statepath=RUN/"state.json"
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    if a.command=="worker":
        log=(RUN/"controller.log").open("a", buffering=1)
        sys.stdout=log; sys.stderr=log
        # OS-held lock prevents duplicate controllers, including simultaneous starts.
        lock=(RUN/"controller.lock").open("a+b");lock.seek(0);lock.write(b"0");lock.flush();lock.seek(0)
        if os.name=="nt":
            import msvcrt;msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:worker(json.loads(a.plan.read_text()),a.resume)
        except Exception as e:
            write(statepath,dict(pid=os.getpid(),status="failed",error=str(e),heartbeat=time.time()));raise
    elif a.command=="check":
        print(json.dumps(validate_plan(json.loads(a.plan.read_text())),indent=2))
    elif a.command in ["start","resume-latest","resume-best"]:
        if alive(state.get("pid")) or alive(state.get("training_pid")):raise RuntimeError("suite or training already running")
        if a.command=="start" and any(RUN.glob("*/latest.pt")):raise RuntimeError("use resume-latest for existing runs")
        plan=json.loads(a.plan.read_text()); snapshot=RUN/"plan.json"
        if snapshot.exists() and json.loads(snapshot.read_text())!=plan:raise RuntimeError("plan changed; use a separate suite directory")
        write(snapshot,plan)
        hashes=RUN/"source_hashes.json"
        if hashes.exists() and json.loads(hashes.read_text())!=source_hashes():raise RuntimeError("source changed; use a separate suite directory")
        write(hashes,source_hashes())
        validate_plan(plan)
        resume="best" if a.command=="resume-best" else "latest"
        log=(RUN/"controller.log").open("a")
        flags=0x00000008|0x00000200|0x08000000 if os.name=="nt" else 0
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--plan",str(snapshot),"--resume",resume,"--run-dir",str(RUN)]
        if os.name=="nt":
            # WMI creates the worker outside OpenSSH's session job; Popen alone is killed on disconnect.
            line=subprocess.list2cmdline(command).replace("'","''")
            ps="$ProgressPreference='SilentlyContinue'; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+line+"'} | ConvertTo-Json -Compress"
            result=subprocess.run(["powershell","-NoProfile","-EncodedCommand",base64.b64encode(ps.encode("utf-16le")).decode()],capture_output=True,text=True,check=True)
            launched=json.loads(result.stdout)
            if launched["ReturnValue"]!=0:raise RuntimeError(f"WMI launch failed: {launched}")
            pid=launched["ProcessId"]
        else:
            pid=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid
        write(statepath,dict(pid=pid,status="starting",heartbeat=time.time()));print(f"controller PID={pid}")
    elif a.command=="status":print(json.dumps(dict(state,controller_alive=alive(state.get("pid")),training_alive=alive(state.get("training_pid"))),indent=2))
    elif a.command=="inspect":
        for path in sorted(RUN.glob("*/history.jsonl")):
            rows=[json.loads(x) for x in path.read_text().splitlines()];print(path.parent.name,json.dumps(min(rows,key=lambda x:x["val_mse"])))
    elif a.command=="logs":
        offsets={}
        while True:
            files=[RUN/"controller.log"]+list(RUN.glob("*/train.log"))
            for path in files:
                if not path.exists():continue
                with path.open(errors="replace") as f:
                    f.seek(offsets.get(str(path),0)); content=f.read(); offsets[str(path)]=f.tell()
                    if content:print(path.parent.name+": "+content,flush=True)
            if not a.follow:break
            time.sleep(2)
if __name__=="__main__":main()
