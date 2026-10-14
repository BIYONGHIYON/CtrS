
import os, sys, json, copy, traceback
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from variable_scale import engine as E
from variable_scale.data import canonical_hash, sha_file
from variable_scale.reporting import finish_job

spec = json.loads(Path(sys.argv[1]).read_text())
run = Path(spec["run_dir"])
source = Path(spec["source_run"])
run.mkdir(parents=True, exist_ok=True)

def compact_history(history):
    out = []
    for row in history:
        row = dict(row)
        row["validation"] = {
            scale: {k: v for k, v in values.items()
                    if k not in ("samples", "scenes")}
            for scale, values in row["validation"].items()
        }
        out.append(row)
    return out

# full RR evaluation remains unchanged in reporting.finish_job.
original_evaluate = E.evaluate
def compact_validation(*args, **kwargs):
    values = original_evaluate(*args, **kwargs)
    return {
        scale: {k: v for k, v in item.items()
                if k not in ("samples", "scenes")}
        for scale, item in values.items()
    }
E.evaluate = compact_validation

try:
    with E.file_lock(run.with_suffix(".lock")):
        fresh_import = not (run / "latest.pt").exists()
        config = copy.deepcopy(spec["config"])

        if fresh_import:
            # Validate original source/config/data/Torch signature using
            # the original Trainer loader before changing the epoch limit.
            old_identity = json.loads((source / "identity.json").read_text())
            source_state = torch.load(
                source / "latest.pt", map_location="cpu", weights_only=False
            )
            if (source_state["epoch"] != 100
                or source_state["next_batch"] != 0
                or source_state["order"] is not None):
                raise ValueError("원본 latest가 완료된 100에포크 경계가 아닙니다.")
            if source_state["signature"] != canonical_hash(old_identity):
                raise ValueError("원본 checkpoint와 identity가 일치하지 않습니다.")

            # Copy only the recovery checkpoint; original remains untouched.
            import shutil
            shutil.copy2(source / "latest.pt", run / "latest.pt")
            trainer = E.Trainer(
                config, spec["job"], spec["manifest"], run,
                spec["device"], spec["source_hash"], resume=True
            )

            trainer.history = compact_history(trainer.history)
            trainer.config = dict(config, epochs=spec["target_epochs"])
            trainer.identity = dict(
                trainer.identity, config=trainer.config
            )
            trainer.signature = canonical_hash(trainer.identity)
            E.json_write(run / "identity.json", trainer.identity)
            E.json_write(run / "parent_checkpoint.json", {
                "path": str(source / "latest.pt"),
                "sha256": sha_file(source / "latest.pt"),
                "original_signature": source_state["signature"],
                "original_epochs": 100,
                "target_epochs": spec["target_epochs"],
                "torch": torch.__version__,
            })
            trainer.save_best()
            trainer.save()
            del source_state
        else:
            # If import was interrupted before migration, stop clearly.
            identity = json.loads((run / "identity.json").read_text())
            if identity["config"]["epochs"] != spec["target_epochs"]:
                raise RuntimeError(
                    "초기 가져오기가 중단되었습니다. 이 새 OUTPUT의 상태를 "
                    "확인한 뒤 다른 OUTPUT 경로로 다시 시작하세요."
                )
            config["epochs"] = spec["target_epochs"]
            trainer = E.Trainer(
                config, spec["job"], spec["manifest"], run,
                spec["device"], spec["source_hash"], resume=True
            )

        (run / "STOP").unlink(missing_ok=True)
        if not trainer.run():
            sys.exit(2)

        # One preselected example per scale: six images across C/D.
        finish_job(trainer, {"indices": [spec["example_index"]]}, False)
        E.json_write(run / "completion.json", {
            "epochs": trainer.epoch,
            "best_epoch": trainer.best_epoch,
            "attempted_updates": trainer.attempted_updates,
            "applied_updates": trainer.applied_updates,
            "exposures": trainer.exposures,
            "scene_split_verified": spec["manifest"]["scene_split_verified"],
            "fr_evaluated": False,
        })
except BaseException as error:
    E.json_write(run / "failure.json", {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    })
    raise
