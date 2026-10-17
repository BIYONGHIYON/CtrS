"""Explicit fresh/resume suites and two independent GPU work queues."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import h5py
import numpy as np
import torch
from .data import canonical_hash, inspect_sources, sha_file
from .engine import KST, Trainer, file_lock, json_write
from .reporting import archive_suite, finish_job, select_examples, suite_summary
from .acquire import resolve_dataset

DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
DEFAULT_PATHS = {"train": f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5",
                 "val": f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5",
                 "RR": f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5",
                 "FR": f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"}
DEFAULT_TRAIN = {"epochs": 100, "batch_size": 32, "micro_batch": 32, "lr": 1e-4,
                 "seed": 42, "amp": False, "deterministic": True,
                 "checkpoint_seconds": 60, "status_seconds": 15, "cpu_threads": 2, "smoke": False}
JOBS = [{"name": "B_x2", "legacy": False, "conditioned": False, "scales": [2]},
        {"name": "B_x3", "legacy": False, "conditioned": False, "scales": [3]},
        {"name": "B_x4", "legacy": False, "conditioned": False, "scales": [4]},
        {"name": "C_mixed", "legacy": False, "conditioned": False, "scales": [2, 3, 4]},
        {"name": "D_mixed", "legacy": False, "conditioned": True, "scales": [2, 3, 4]}]

def source_files():
    root = Path(__file__).resolve().parent
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.suffix in (".py", ".json")}

def source_identity(files):
    import hashlib
    return canonical_hash({name: hashlib.sha256(value).hexdigest() for name, value in files.items()})

def smoke_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(9214)
    provenance, paths = {}, {}
    for split, count in [("train", 6), ("val", 3), ("RR", 5), ("FR", 5)]:
        gt = rng.uniform(10, 2000, (count, 4, 24, 24)).astype(np.float32)
        pan = gt.mean(axis=1, keepdims=True)
        ms = gt.reshape(count, 4, 6, 4, 6, 4).mean((3, 5))
        lms = np.repeat(np.repeat(ms, 4, axis=2), 4, axis=3)
        path = root / f"{split}.h5"
        with h5py.File(path, "w") as f:
            for key, value in {"pan": pan, "ms": ms, "lms": lms}.items():
                f.create_dataset(key, data=value)
            if split != "FR":
                f.create_dataset("gt", data=gt)
        paths[split] = str(path.resolve())
        provenance[split] = [{"scene_id": f"fixture-{split}-{i}", "tile": "artificial 24x24 fixture"}
                             for i in range(count)]
    scene_path = root / "scene_manifest.json"
    json_write(scene_path, provenance)
    return paths, str(scene_path)

def validate_training(config):
    if (config["epochs"] < 1 or config["batch_size"] < 1 or config["micro_batch"] < 1
        or config["micro_batch"] > config["batch_size"] or config["lr"] <= 0
        or config["checkpoint_seconds"] <= 0 or config["status_seconds"] <= 0):
        raise ValueError("Invalid training configuration")

def prepare_output(output, files, resume=False, resume_input=None):
    output = Path(output).resolve()
    source = Path(resume_input).resolve() if resume_input else None
    if source:
        if not resume or not (source/"suite_identity.json").is_file():
            raise ValueError("Saved Output resume needs explicit resume and a suite_identity.json")
        if output.exists():
            raise FileExistsError("Import Saved Output into a fresh OUTPUT_DIR")
        shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.tmp", "logs"))
    elif not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output/"suite_identity.json").is_file():
        raise FileNotFoundError("Explicit resume requested but existing suite not found")
    if resume:
        expected = json.loads((output/"suite_identity.json").read_text(encoding="utf-8"))
        if expected["source_hash"] != source_identity(files):
            raise ValueError("Code changed; a new experiment directory is required")
    code = output / "code" / "variable_scale"
    code.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = code/name
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Source snapshot mismatch")
        if not target.exists():
            target.write_bytes(content)
    return output

def worker(spec):
    run = Path(spec["run_dir"])
    with file_lock(run.with_suffix(".lock")):
        return _worker(spec)

def _worker(spec):
    os.environ["CTRS_INVOCATION_ID"] = spec["invocation_id"]
    run = Path(spec["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    try:
        trainer = Trainer(spec["config"], spec["job"], spec["manifest"], run,
                          spec["device"], spec["source_hash"], spec["resume"])
        if spec.get("evaluate_only"):
            if trainer.epoch != spec["config"]["epochs"] or trainer.best_model is None:
                raise ValueError("Evaluation requires a fully trained matching checkpoint")
        elif not trainer.run():
            return 2
        finish_job(trainer, spec["selection"], spec["provisional_fr"])
        return 0
    except BaseException as error:
        import traceback
        log_dir = run / "logs" / spec["invocation_id"]
        json_write(log_dir / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc()})
        raise

def dashboard(queues, active, completed, output, suite_started, invocation_id):
    lines = ["SSA-MRN 배율별 노출량 통제 | 이번 실행만 표시 | " + dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")]
    for device, queue in queues.items():
        state = active.get(device)
        if state:
            job, process, _ = state
            status_path = output / "runs" / job["name"] / "logs" / invocation_id / "status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                status = None
            if status and status.get("invocation_id") == invocation_id and status.get("job") == job["name"]:
                mem = status["memory_bytes"]
                lines.append(f"{device} {job['name']} | {status['stage']} | "
                             f"epoch {status['epoch']}/{status['epochs']} | batch {status['batch']}/{status['batches']} | "
                             f"에포크 {status['epoch_percent']:.1f}% / 학습 {status['total_percent']:.1f}%")
                loss = status["train_mse"]
                lines.append(f"  lr={status['lr']:.3g} loss={loss if loss is not None else '측정 대기'} | "
                             f"val(current epoch)={status['current_validation']}")
                lines.append("  GPU allocated/reserved/peak=" + "/".join(f"{mem[k]/1024**3:.2f}GiB" for k in ("allocated","reserved","peak")))
                remaining = status["eta_training_seconds"]
                lines.append(f"  남은 학습 시간: {str(dt.timedelta(seconds=round(remaining))) if remaining is not None else '측정 대기'} | "
                             f"학습 종료 예상: {status['eta_training_kst']}")
                lines.append("  전체 종료 예상: " + status["eta_overall"])
                lines.append("  ETA 제외 범위: " + ", ".join(status["eta_exclusions"]))
                lines.append(f"  update 이번 호출={status['updates_this_invocation']} / 누적={status['updates_total']}")
            else:
                lines.append(f"{device} {job['name']} | 시작/데이터 GPU 상주 준비 | ETA 측정 대기")
        else:
            lines.append(f"{device} | idle")
        # Pending models have not been measured: do not assign invented durations.
        lines.append("  GPU 대기 큐: " + (", ".join(j["name"] for j in queue) or "없음") + " | 큐 포함 종료 예상: 미측정 작업이 있으면 측정 대기")
    lines.append(f"suite 완료 작업 {len(completed)} | elapsed {time.monotonic()-suite_started:.1f}s")
    return "\n".join(lines)

def run_suite(settings, notebook=False):
    with file_lock(Path(settings["output_dir"]).resolve().with_suffix(".lock")):
        return _run_suite(settings, notebook)

def _run_suite(settings, notebook=False):
    mode = settings.get("mode", "smoke")
    if mode not in ("smoke", "train", "evaluate"):
        raise ValueError("mode must be smoke/train/evaluate")
    output_path = settings["output_dir"]
    resume = settings.get("resume", False) or mode == "evaluate"
    if mode == "smoke" and resume:
        raise ValueError("Smoke suites are fresh-only; use train/evaluate for real resumable runs")
    config = dict(DEFAULT_TRAIN, **settings.get("training", {}))
    config["smoke"] = mode == "smoke"
    if mode == "smoke":
        config.update(epochs=2, batch_size=2, micro_batch=2, amp=False, status_seconds=.5)
    validate_training(config)
    devices = settings.get("devices", ["cuda:0", "cuda:1"])
    if devices == ["cpu"]:
        if mode != "smoke":
            raise ValueError("CPU is explicit local smoke only; no training fallback")
    else:
        if devices != ["cuda:0", "cuda:1"] or torch.cuda.device_count() < 2:
            raise RuntimeError("Requires two CUDA GPUs (T4 x2); no silent CPU fallback")
        props = [torch.cuda.get_device_properties(i) for i in (0,1)]
        if (props[0].name, props[0].total_memory) != (props[1].name, props[1].total_memory):
            raise RuntimeError("Paired GPU comparison requires equivalent devices")
    files = source_files()
    output = prepare_output(output_path, files, resume, settings.get("resume_input"))
    invocation_id = uuid.uuid4().hex
    invocation_dir = output / "logs" / invocation_id
    invocation_dir.mkdir(parents=True, exist_ok=False)
    def event(stage, **values):
        row = dict(values, stage=stage, invocation_id=invocation_id,
                   timestamp_kst=dt.datetime.now(KST).isoformat())
        with (invocation_dir / "controller.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        detail = ""
        if "split" in values:
            detail += f" | {values['split']}"
        if "done" in values:
            done, total = values["done"], values["total"]
            detail += f" | {done:,}/{total:,} {values['unit']} ({100*done/max(1,total):.1f}%)"
        if "elapsed_seconds" in values:
            detail += f" | {values['elapsed_seconds']:.1f}s"
        if "path" in values:
            detail += f" | {values['path']}"
        print(f"[현재 실행] {stage}{detail}", flush=True)
    if notebook:
        from IPython.display import clear_output
        clear_output(wait=True)
    event("start")
    if mode == "smoke":
        paths, provenance = smoke_data(output / "fixture_data")
    else:
        paths, acquisition = resolve_dataset(settings.get("paths", DEFAULT_PATHS),
                require_fr=settings.get("provisional_fr", False),
                version=settings.get("dataset_version"), cache=settings.get("data_cache"), event=event)
        json_write(invocation_dir / "dataset_acquisition.json", acquisition)
        provenance = settings.get("scene_manifest")
    event("dataset_validation")
    manifest = inspect_sources(paths, provenance, settings.get("allow_unverified_splits", False), event=event)
    selection = select_examples(manifest)
    names = settings.get("job_names", [job["name"] for job in JOBS])
    if not names or len(set(names)) != len(names) or set(names)-{job["name"] for job in JOBS}:
        raise ValueError("Select unique job names from the six defined A/B/C/D arms")
    jobs = [job for job in JOBS if job["name"] in names]
    identity_data = json.loads(json.dumps(manifest))
    for item in identity_data["files"].values():
        item.pop("path", None)
    identity = {"source_hash": source_identity(files), "config": config, "jobs": jobs,
                "data": identity_data, "selection": selection,
                "provisional_fr": settings.get("provisional_fr", False)}
    if resume and json.loads((output/"suite_identity.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("Suite config/data/selection changed: use a new OUTPUT_DIR")
    json_write(output / "suite_identity.json", identity)
    json_write(output / "data_manifest.json", manifest)
    json_write(output / "selection.json", selection)
    json_write(output / "settings.json", settings)
    queues = {device: jobs[i::len(devices)] for i, device in enumerate(devices)}
    json_write(output / "gpu_queues.json", {k: [j["name"] for j in v] for k,v in queues.items()})
    active, completed = {}, []
    suite_started = time.monotonic()
    display_handle = None
    env = dict(os.environ, PYTHONPATH=str(output/"code"), PYTHONIOENCODING="utf-8",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", CTRS_INVOCATION_ID=invocation_id)
    failure = None
    try:
        while any(queues.values()) or active:
            for device, queue in queues.items():
                if device not in active and queue:
                    job = queue.pop(0)
                    run = output / "runs" / job["name"]
                    run.mkdir(parents=True, exist_ok=True)
                    latest_exists = (run/"latest.pt").is_file()
                    if resume and not latest_exists and (run/"identity.json").exists():
                        raise FileNotFoundError(f"Missing latest for already-started job {job['name']}")
                    # Clear this parent's prior cooperative stop before spawning, not
                    # inside the child where a fresh parent stop could be lost.
                    (run/"STOP").unlink(missing_ok=True)
                    # Unstarted queue entries in a valid resumed suite legitimately start fresh.
                    spec = {"invocation_id": invocation_id,
                            "config": config, "job": job, "manifest": manifest, "run_dir": str(run),
                            "device": device, "source_hash": identity["source_hash"],
                            "resume": resume and latest_exists, "selection": selection,
                            "evaluate_only": mode == "evaluate", "provisional_fr": identity["provisional_fr"]}
                    current_logs = run / "logs" / invocation_id
                    current_logs.mkdir(parents=True, exist_ok=True)
                    spec_path = current_logs / "worker_spec.json"
                    json_write(spec_path, spec)
                    log = (current_logs/"worker.log").open("x", encoding="utf-8")
                    process = subprocess.Popen([sys.executable, "-m", "variable_scale.runner", "--worker", str(spec_path)],
                                               stdout=log, stderr=subprocess.STDOUT, env=env)
                    active[device] = (job, process, log)
            for device, (job, process, log) in list(active.items()):
                result = process.poll()
                if result is not None:
                    log.close()
                    del active[device]
                    if result != 0:
                        failure_path = output/"runs"/job["name"]/"logs"/invocation_id/"failure.json"
                        detail = failure_path.read_text(encoding="utf-8") if failure_path.exists() else f"worker exit={result}; inspect worker.log"
                        failure = RuntimeError(f"{device} {job['name']} failed: {detail}")
                        raise failure
                    completed.append(job["name"])
            text = dashboard(queues, active, completed, output, suite_started, invocation_id)
            if notebook:
                from IPython.display import display
                if display_handle is None:
                    display_handle = display({"text/plain": text}, raw=True, display_id=True)
                else:
                    display_handle.update({"text/plain": text}, raw=True)
            else:
                # Overwrite a terminal screen; noninteractive runs can use the status JSON directly.
                print("\033[2J\033[H" + text, end="", flush=True)
            time.sleep(1 if mode == "smoke" else 5)
        suite_summary(output, jobs)
        archive = archive_suite(output, invocation_id)
        json_write(output/"execution_status.json", {"implementation_run_complete": True,
                   "smoke_only": mode == "smoke", "training_complete": mode != "smoke",
                   "validated_fr_complete": False, "remote_archive_verified": False,
                   "finished_kst": dt.datetime.now(KST).isoformat(),
                   "suite_seconds": time.monotonic()-suite_started, "archive": str(archive)})
        # Include final execution status in the archive and reverify checksums.
        archive = archive_suite(output, invocation_id)
        final_text = (f"{'Smoke 기능 검증' if mode == 'smoke' else '학습·RR 실행'} 완료: {output}\n"
                      f"ZIP: {archive}\nFR 지표 검증·원격 보관 검증은 미완료. "
                      "smoke 수치는 성능 개선 근거가 아닙니다.")
        if display_handle:
            display_handle.update({"text/plain": final_text}, raw=True)
        else:
            print("\n" + final_text)
        return output
    finally:
        for job, process, _ in active.values():
            if process.poll() is None:
                # Portable cooperative stop: Windows Popen does not support SIGINT.
                (output/"runs"/job["name"]/"STOP").touch()
        # Workers save at the next safe boundary; parent never force-kills a valid checkpoint.
        for _, process, log in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            log.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["smoke", "train", "evaluate"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--jobs", nargs="+", choices=[job["name"] for job in JOBS])
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(json.loads(args.worker.read_text(encoding="utf-8"))))
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.mode:
        settings["mode"] = args.mode
    if args.output:
        settings["output_dir"] = str(args.output.resolve())
    if args.device:
        settings["devices"] = ["cpu"] if args.device == "cpu" else ["cuda:0", "cuda:1"]
    if args.jobs:
        settings["job_names"] = args.jobs
    if "output_dir" not in settings:
        parser.error("--output or config output_dir is required")
    run_suite(settings)

if __name__ == "__main__":
    main()
