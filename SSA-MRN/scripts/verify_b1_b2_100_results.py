"""Verify saved B1/B2 100-epoch evidence and its interrupted 200-cell provenance.

This checks saved evidence without training, reading H5, or executing pickle code.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from verify_kaggle_result import checkpoint_epoch, read, sha


def check_png(path, expected_size=None):
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", path
    dimensions = struct.unpack(">II", header[16:24])
    if expected_size:
        assert dimensions == expected_size, (path, dimensions)
    return dimensions


def audit(root, recovery, cell=None):
    root, recovery = root.resolve(), recovery.resolve()
    source_manifest = read(root / "artifact_manifest.json")["artifacts"]
    for relative, expected in source_manifest.items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert sha(path) == expected["sha256"], relative
    results = read(root / "results.json")
    assert set(results) == {f"{v}_QB_{v}_k6_s42" for v in ("B1", "B2")}
    data = read(root / "data_manifest.json")
    selection = read(root / "selection.json")
    selected = sorted(np.random.default_rng(42).choice(20, 5, replace=False).tolist())
    assert selection["zero_based_scene_indices"] == selected == [1, 8, 11, 13, 19]
    assert selection["tiles"] == "full scene"
    assert read(root / "suite_config.json")["epochs"] == 100
    report = {"verified_manifest_files": len(source_manifest), "runs": {},
              "checkpoint_scope": "ZIP CRC, metadata epoch and byte hashes; tensors not loaded",
              "limitations": ["No local training or H5/GT metric recomputation.",
                              "Saved predictions checked for shape and finite values; no forward inference rerun.",
                              "FR QNR/Ds and Q2n/SCC MATLAB parity remains unverified."]}
    histories, initializations = [], []
    for run_id, result in results.items():
        run, old_run = root / "runs" / run_id, recovery / "runs" / run_id
        history = read(run / "history.json")
        assert [row["epoch"] for row in history] == list(range(1, 101))
        for row in history:
            assert row["lr"] == 1e-4 and row["micro_batch"] == 32
            assert all(math.isfinite(row[key]) for key in
                       ("train_mse", "val_mse", "val_psnr_peak1", "val_sam_deg", "seconds"))
        best = min(history, key=lambda row: row["val_mse"])
        assert (best["epoch"], best["val_mse"]) == (result["best_epoch"], result["best_mse"])
        config = result["config"]
        assert config["epochs"] == 100 and config["schedule"] == "constant"
        assert config["seed"] == 42 and config["k"] == 6 and config["scale"] == 2047.
        assert config["amp"] and config["controlled"] and config["channels_last"]
        assert config["fused_adam"] and config["fft_observation"] and config["gpu_cache"]
        assert config["batch_size"] == config["micro_batch"] == 32 and config["workers"] == 0
        assert sha(root / "code/kaggle_worker.py") == config["worker_sha256"]
        assert sha(root / "code/B2_models.py") == config["B2_source_sha256"]
        assert sha(root / "observation_train.json") == config["observation_profile_sha256"]
        raw_result = read(run / "results.json")
        assert set(result) - set(raw_result) == {"delta_from_same_seed_B1"}
        assert raw_result == {key: result[key] for key in raw_result}
        done = read(run / "training_complete.json")
        assert done["config"] == config and done["epoch"] == 100
        assert done["best_epoch"] == best["epoch"] and done["best_mse"] == best["val_mse"]
        checkpoints = {}
        for kind in ("best", "latest"):
            path = run / f"{kind}.pt"
            epoch = best["epoch"] if kind == "best" else 100
            assert checkpoint_epoch(path) == epoch
            assert sha(path) == result["checkpoints"][kind]["sha256"]
            checkpoints[kind] = {"epoch": epoch, "sha256": sha(path), "bytes": path.stat().st_size}
        previous_epoch = checkpoint_epoch(run / "latest.prev.pt")
        assert 0 <= previous_epoch <= 100
        imported = read(run / "bootstrap_manifest.json")
        old_history = read(old_run / "history.json")
        imported_epoch = imported["completed_epochs"]
        assert not imported["partial_epoch"] and imported_epoch == imported["history_imported"]
        assert imported_epoch == len(old_history) == checkpoint_epoch(old_run / "latest.pt")
        assert history[:imported_epoch] == old_history
        assert sha(old_run / "latest.pt") == imported["original_sha256"]
        assert imported["original_config"]["epochs"] == 200
        assert sha(recovery / "code/kaggle_worker.py") == imported["original_config"]["worker_sha256"]
        assert sha(recovery / "code/B2_models.py") == config["B2_source_sha256"]
        for split in ("train", "val", "RR", "FR"):
            assert config["data"][split] == data[split]
            assert imported["original_config"]["data"][split] == data[split]
        assert read(run / "bootstrap_initialization.json") == read(old_run / "initialization.json")
        assert read(run / "initialization.json") == read(old_run / "initialization.json")
        for protocol in ("RR", "FR"):
            assert config["data"][protocol] == data[protocol]
            block = result["metrics"][protocol]
            assert block["count"] == data[protocol]["shapes"]["pan"][0] == 20
            rows = block["rows"]
            assert [row["scene_index"] for row in rows] == list(range(20))
            assert rows == read(run / f"{protocol}_per_scene.json")
            for key, value in block["mean"].items():
                assert all(math.isfinite(row[key]) for row in rows)
                assert math.isclose(value, np.mean([row[key] for row in rows]), rel_tol=1e-12)
            predictions = sorted((run / "predictions" / protocol).glob("*.npz"))
            assert len(predictions) == 20
            for index, path in enumerate(predictions):
                with np.load(path, allow_pickle=False) as saved:
                    assert int(saved["scene_index"]) == index and float(saved["scale"]) == 2047.
                    assert saved["prediction_dn"].shape == tuple(data[protocol]["shapes"]["lms"][1:])
                    assert np.isfinite(saved["prediction_dn"]).all() and "gt" not in saved.files
        assert result["full_scene"] and result["test_precision"] == "FP32"
        assert [image["scene_index"] for image in result["images"]] == selected
        for image in result["images"]:
            check_png(run / image["file"], (1024, 296))
        evaluation = read(run / "evaluation_complete.json")
        assert evaluation["RR"] == evaluation["FR"] == 20
        assert evaluation["best_sha256"] == checkpoints["best"]["sha256"]
        for kind in ("best", "latest"):
            checkpoint_epoch(old_run / f"{kind}.pt")
        report["runs"][run_id] = {"completed_epochs": 100, "best_epoch": best["epoch"],
                                 "best_validation_mse": best["val_mse"], "checkpoints": checkpoints,
                                 "imported_epochs": imported_epoch,
                                 "source_checkpoint_sha256": imported["original_sha256"],
                                 "RR": result["metrics"]["RR"]["mean"],
                                 "FR": result["metrics"]["FR"]["mean"],
                                 "prediction_arrays": 40,
                                 "epoch_seconds": sum(row["seconds"] for row in history),
                                 "this_session": read(run / "session_timing.json")}
        histories.append([row["sample_order_sha256"] for row in history])
        initializations.append(read(run / "initialization.json")["core_sha256"])
    assert histories[0] == histories[1] and initializations[0] == initializations[1]
    assert results["B1_QB_B1_k6_s42"]["images"] == results["B2_QB_B2_k6_s42"]["images"]
    report["paired_initialization_and_100_epoch_orders_verified"] = True
    paired = read(root / "paired_comparison.json")[0]["metric_delta_B2_minus_B1"]
    b1, b2 = results["B1_QB_B1_k6_s42"], results["B2_QB_B2_k6_s42"]
    for protocol, metrics in paired.items():
        for key, value in metrics.items():
            assert math.isclose(value, b2["metrics"][protocol]["mean"][key] -
                                b1["metrics"][protocol]["mean"][key], abs_tol=1e-14)
    report["B2_better_RR_PSNR_scenes"] = sum(b["PSNR"] > a["PSNR"] for a, b in zip(
        b1["metrics"]["RR"]["rows"], b2["metrics"]["RR"]["rows"]))
    for name in ("learning_seed42.png", "test_comparison_seed42.png"):
        check_png(root / name)
    if cell:
        tree = ast.parse(cell.read_text(encoding="utf-8"))
        embedded = {node.targets[0].id: node.value.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
        for variable, filename in (("WORKER", "kaggle_worker.py"), ("PASTED_MODEL_SOURCE", "B2_models.py")):
            ast.parse(embedded[variable])
            assert hashlib.sha256(embedded[variable].encode()).hexdigest() == sha(root / "code" / filename)
        report["reproduction_cell_worker_and_model_match_saved_runtime"] = True
    report["status"] = "saved evidence verified; original tensors and H5 not re-evaluated"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.input, args.recovery, args.cell)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
