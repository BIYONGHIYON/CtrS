"""Audit saved Kaggle evidence without executing pickled checkpoint code."""
import argparse
import ast
import hashlib
import json
import math
import pickletools
from pathlib import Path
import zipfile

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_epoch(path):
    # torch.save stores metadata in a pickle inside a ZIP. Inspect opcodes,
    # never unpickle an untrusted checkpoint merely to inspect its epoch.
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, f"Invalid checkpoint ZIP: {path}"
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        assert len(names) == 1
        operations = list(pickletools.genops(archive.read(names[0])))
        for index, (opcode, value, _) in enumerate(operations):
            if value != "epoch":
                continue
            for following, scalar, _ in operations[index + 1:]:
                if following.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT"):
                    continue
                assert following.name in ("BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4")
                return int(scalar)
    raise ValueError(f"Checkpoint epoch not found: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = read(root / "artifact_manifest.json")
    for relative, expected in manifest["artifacts"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root), f"Unsafe manifest path: {relative}"
        assert path.stat().st_size == expected["bytes"], relative
        assert sha(path) == expected["sha256"], relative

    results = read(root / "results.json")
    assert set(results) == {"github_QB_baseline_k6_s42", "pasted_QB_band_gated_hf_k6_s42"}
    data = read(root / "data_manifest.json")
    selection = read(root / "selection.json")
    expected_selection = sorted(np.random.default_rng(42).choice(20, 5, replace=False).tolist())
    assert selection["zero_based_scene_indices"] == expected_selection
    assert selection["tiles"] == "full scene"
    report = {"input_manifest_verified_files": len(manifest["artifacts"]), "runs": {},
              "limitations": ["Original H5 data is external; metrics were not recomputed against GT locally.",
                             "Checkpoint tensors were not loaded with PyTorch; ZIP CRC, metadata epoch and hashes were checked."]}
    orders, core_hashes = [], []
    for run_id, result in results.items():
        run = root / "runs" / run_id
        history = read(run / "history.json")
        assert [row["epoch"] for row in history] == list(range(1, 101))
        assert all(row["micro_batch"] == 32 and all(math.isfinite(row[key]) for key in
                   ("train_mse", "val_mse", "val_psnr_peak1", "val_sam_deg", "seconds")) for row in history)
        best = min(history, key=lambda row: row["val_mse"])
        assert best["epoch"] == result["best_epoch"]
        assert best["val_mse"] == result["best_mse"]
        config = result["config"]
        assert config["amp"] and config["controlled"] and config["channels_last"]
        assert config["batch_size"] == config["micro_batch"] == 32
        assert config["seed"] == 42 and config["k"] == 6 and config["lr"] == 1e-4
        assert sha(root / "code/kaggle_worker.py") == config["worker_sha256"]
        assert sha(root / "code/pasted_models.py") == config["pasted_source_sha256"]
        for kind, expected in result["checkpoints"].items():
            checkpoint = run / f"{kind}.pt"
            epoch = result["best_epoch"] if kind == "best" else 100
            assert checkpoint_epoch(checkpoint) == epoch
            assert sha(checkpoint) == expected["sha256"]
        orders.append([row["sample_order_sha256"] for row in history])
        core_hashes.append(read(run / "initialization.json")["core_sha256"])
        for protocol in ("RR", "FR"):
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
                    assert int(saved["scene_index"]) == index
                    assert float(saved["scale"]) == 2047.
                    assert saved["prediction_dn"].shape == tuple(data[protocol]["shapes"]["lms"][1:])
                    assert np.isfinite(saved["prediction_dn"]).all()
                    assert "gt" not in saved.files
        assert [item["scene_index"] for item in result["images"]] == expected_selection
        for image in result["images"]:
            path = run / image["file"]
            assert path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert read(run / "evaluation_complete.json")["best_sha256"] == result["checkpoints"]["best"]["sha256"]
        report["runs"][run_id] = {"epochs": 100, "best_epoch": result["best_epoch"],
                                 "RR_scenes": 20, "FR_scenes": 20, "prediction_arrays": 40,
                                 "RR_mean": result["metrics"]["RR"]["mean"],
                                 "FR_mean": result["metrics"]["FR"]["mean"]}
    assert orders[0] == orders[1] and core_hashes[0] == core_hashes[1]
    report["paired_initialization_and_epoch_orders_verified"] = True
    paired = read(root / "paired_comparison.json")[0]["metric_delta_pasted_minus_github"]
    old, new = results["github_QB_baseline_k6_s42"], results["pasted_QB_band_gated_hf_k6_s42"]
    for protocol, metrics in paired.items():
        for key, value in metrics.items():
            assert math.isclose(value, new["metrics"][protocol]["mean"][key] -
                                old["metrics"][protocol]["mean"][key], abs_tol=1e-14)
    if args.cell:
        tree = ast.parse(args.cell.read_text(encoding="utf-8"))
        worker = next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "WORKER" for target in node.targets))
        ast.parse(worker)
        assert hashlib.sha256(worker.encode()).hexdigest() == sha(root / "code/kaggle_worker.py")
        report["reproduction_cell_worker_matches_executed_worker"] = True
    report["status"] = "verified saved evidence; not a local training/evaluation rerun"
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
