import json
import subprocess
from datetime import datetime
from pathlib import Path
import yaml

root = Path(r"C:\CtrS-ecrformer-winter\ECRformer\Official_ECRformer")
hp = root / "experiments/ecrformer_spring_winter_3000_each_seed42_20ep/version_1/hparams.yaml"
cfg = yaml.load(hp.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["config"]
ckpt = hp.parent / "checkpoints/last.ckpt"
assert ckpt.is_file()
args = [r"C:\CtrS\.venv\Scripts\python.exe", "-u", "resume_winter_exact.py"]
status = root / "winter_resume_status.json"
with (root / "winter_train.log").open("a", encoding="utf-8", buffering=1) as log:
    log.write("\n[RESUME_LAUNCH] " + datetime.now().astimezone().isoformat() + "\n")
    process = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    record = {"pid": process.pid, "started": datetime.now().astimezone().isoformat(),
              "checkpoint": str(ckpt), "args": args}
    status.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    code = process.wait()
    record.update(exit_code=code, ended=datetime.now().astimezone().isoformat())
    status.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    log.write("\n[PROCESS_EXIT] " + json.dumps(record, ensure_ascii=True) + "\n")
