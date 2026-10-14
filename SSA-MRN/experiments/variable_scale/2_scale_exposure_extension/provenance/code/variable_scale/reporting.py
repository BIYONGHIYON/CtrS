"""Actual curves, fixed panels, timing measurements, manifests and archive."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ResidentSplit, sha_file
from .engine import evaluate, json_write, predict, synchronize
from .metrics_snapshot import fr_metrics

def select_examples(manifest, count=5, seed=42):
    entries = manifest["files"]["RR"]["samples"]
    representatives = {}
    for i, entry in enumerate(entries):
        representatives.setdefault(entry["scene_id"], i)
    if len(representatives) < count:
        raise ValueError(f"RR needs {count} distinct scenes (or explicitly unverified samples)")
    candidates = sorted(representatives.values())
    ids = sorted(np.random.default_rng(seed).choice(candidates, count, replace=False).tolist())
    return {"seed": seed, "indices": ids, "samples": [dict(entries[i], index=i) for i in ids],
            "rule": "one first sample per scene, seeded choice then ascending index",
            "scene_identity_verified": manifest["files"]["RR"]["scene_provenance_present"],
            "tile": "whole supplied H5 sample; synthetic target uses documented common crop",
            "display_order": ["LR MS", "PAN input", "prediction", "GT"],
            "rgb_bands": [2, 1, 0], "fixed_before_training_and_test_scoring": True}

def panel(batch, prediction, path, title):
    gt = batch["gt"][0].float().cpu().numpy()
    ms = batch["ms"][0:1]
    nearest = torch.nn.functional.interpolate(ms, size=gt.shape[-2:], mode="nearest")[0].cpu().numpy()
    pred = prediction[0].float().cpu().numpy()
    guide = batch["pan"][0, 0].float().cpu().numpy()
    low, high = np.percentile(gt[[2, 1, 0]], [2, 98])
    def rgb(x):
        return np.clip((x[[2, 1, 0]].transpose(1, 2, 0)-low)/max(high-low, 1e-8), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, value, label in zip(axes, [rgb(nearest), guide, rgb(pred), rgb(gt)],
                               ["LR MS (nearest)", "PAN input", "prediction", "GT"]):
        ax.imshow(value, cmap="gray" if label == "PAN input" else None)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    np.save(Path(path).with_suffix(".prediction.npy"), pred)

def history_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_mse"] for r in history], label="train")
    for scale in history[0]["validation"]:
        for ax, metric in zip(axes, ["MSE", "PSNR", "SAM"]):
            ax.plot(epochs, [r["validation"][scale]["scene_mean"][metric] for r in history], label=f"val x{scale}")
    for ax, metric in zip(axes, ["MSE", "PSNR", "SAM"]):
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def benchmark(model, batch, legacy, device, amp, repetitions=5):
    model.eval()
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=amp):
        for _ in range(3):
            predict(model, batch, legacy)
        synchronize(device)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            predict(model, batch, legacy)
        synchronize(device)
        elapsed = (time.perf_counter()-start)/repetitions
    return {"seconds_per_forward": elapsed, "batch_size": len(batch["pan"]),
            "target_shape": list(batch["pan"].shape), "ms_shape": list(batch["ms"].shape),
            "warmup": 3, "repetitions": repetitions, "cuda_synchronized": torch.device(device).type == "cuda",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
            "memory_includes_resident_training_data": True}

def finish_job(trainer, selection, provisional_fr=False):
    output, model, job = trainer.run_dir, trainer.model, trainer.job
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["signature"] != trainer.signature:
        raise ValueError("Best checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    trainer.status("RR scoring")
    begin = time.monotonic()
    rr = ResidentSplit(trainer.manifest["files"]["RR"], trainer.device)
    result = evaluate(model, rr, job["scales"], job["legacy"], trainer.config["micro_batch"], trainer.config["amp"])
    rr_seconds = time.monotonic()-begin
    json_write(output / "rr_metrics.json", {"protocol": "native_x4" if job["legacy"] else "synthetic_proxy",
               "scene_split_verified": trainer.manifest["scene_split_verified"],
               "prediction_clipping": "none", "border_exclusion": 0, "psnr_peak": 1,
               "aggregation": "sample metrics -> scene mean -> equally weighted scene mean", "by_scale": result})
    measurements = {}
    with torch.inference_mode():
        for scale in job["scales"]:
            batch = rr.batch(torch.arange(1, device=rr.device), scale, job["legacy"])
            measurements[str(scale)] = benchmark(model, batch, job["legacy"], rr.device, trainer.config["amp"])
            for idx in selection["indices"]:
                sample = rr.batch(torch.tensor([idx], device=rr.device), scale, job["legacy"])
                with torch.autocast(rr.device.type, enabled=trainer.config["amp"]):
                    pred = predict(model, sample, job["legacy"])
                panel(sample, pred, output / f"rr_x{scale}_sample_{idx:03d}.png",
                      f"{job['name']} x{scale}, sample {idx}; display RGB bands assumed [2,1,0]")
    trainer.status("FR assessment")
    fr_result = {"status": "not_evaluated", "reason": "QNR/D_s MATLAB parity not verified; "
                 "only actual native x4 is eligible. No GT/PSNR is fabricated.", "ratio": 4,
                 "unseen_training_ratio": 4 not in job["scales"]}
    if provisional_fr:
        fr = ResidentSplit(trainer.manifest["files"]["FR"], trainer.device)
        rows = []
        with torch.inference_mode():
            for i in range(fr.length):
                ids = torch.tensor([i], device=fr.device)
                # Actual native sensor observations, never synthetic GT-based FR inputs.
                batch = {k: v.index_select(0, ids) for k, v in fr.tensors.items()}
                batch["scale"] = 4
                with torch.autocast(fr.device.type, enabled=trainer.config["amp"]):
                    prediction = predict(model, batch, job["legacy"])
                arrays = {k: v[0].float().cpu().numpy()*2047 for k, v in batch.items() if isinstance(v, torch.Tensor)}
                metrics = fr_metrics(prediction[0].float().cpu().numpy()*2047,
                                     arrays["lms"], arrays["ms"], arrays["pan"], ratio=4)
                rows.append({"sample_index": i, "scene_id": fr.entries[i]["scene_id"], "metrics": metrics})
        fr_result.update(status="provisional_full_x4", metrics=scene_means(rows),
                         reason="Existing QNR implementation; MATLAB parity UNVERIFIED")
    json_write(output / "fr_metrics.json", fr_result)
    trainer.status("plots and manifest")
    history_plot(trainer.history, output / "learning.png")
    json_write(output / "costs.json", {"parameters": sum(p.numel() for p in model.parameters()),
               "device": str(trainer.device), "device_name": torch.cuda.get_device_name(trainer.device)
                          if trainer.device.type == "cuda" else "CPU explicit smoke",
               "torch": torch.__version__, "amp": trainer.config["amp"], "timing": trainer.timing,
               "rr_evaluation_seconds": rr_seconds, "inference": measurements,
               "optimizer_attempts": trainer.attempted_updates, "optimizer_applied": trainer.applied_updates,
               "effective_batch": trainer.config["batch_size"], "micro_batch": trainer.config["micro_batch"],
               "exposures": trainer.exposures,
               "checkpoint_bytes": {name: (output/name).stat().st_size for name in ("best.pt", "latest.pt")}})
    hashes = {p.name: sha_file(p) for p in output.iterdir() if p.is_file() and p.name not in
              ("artifact_manifest.json", "status.json", "progress.jsonl")}
    json_write(output / "artifact_manifest.json", {"hashes": hashes, "best_epoch": trainer.best_epoch,
               "latest_complete_epoch": trainer.epoch, "code_config_data_signature": trainer.signature,
               "training_complete": True, "rr_complete": True,
               "fr_validated_complete": False, "remote_storage_verified": False,
               "smoke_only": trainer.config.get("smoke", False)})
    trainer.status("complete_for_requested_scope")

def suite_summary(output, jobs):
    output = Path(output)
    compare, init_hashes, order_hashes, costs = [], {}, {}, {}
    for job in jobs:
        run = output / "runs" / job["name"]
        init_hashes[job["name"]] = json.loads((run/"initialization.json").read_text())["common_parameter_sha256"]
        order_hashes[job["name"]] = [r["order_sha256"] for r in json.loads((run/"batch_orders.json").read_text())]
        costs[job["name"]] = json.loads((run/"costs.json").read_text())
        for scale, data in json.loads((run/"rr_metrics.json").read_text())["by_scale"].items():
            if not job["legacy"]:
                compare.append({"job": job["name"], "scale": int(scale), "metrics": data["scene_mean"]})
    rows = []
    for row in compare:
        baseline = next((r for r in compare if r["job"] == f"B_x{row['scale']}" and r["scale"] == row["scale"]), None)
        rows.append(dict(row, baseline=baseline["job"] if baseline else None,
                         delta={k: row["metrics"][k]-baseline["metrics"][k] for k in row["metrics"]} if baseline else None))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, ["PSNR", "SAM", "MSE"]):
        for name in sorted({r["job"] for r in compare}):
            points = sorted([r for r in compare if r["job"] == name], key=lambda r: r["scale"])
            ax.plot([r["scale"] for r in points], [r["metrics"][metric] for r in points], "o-", label=name)
        ax.set_title(f"Synthetic RR {metric}")
        ax.set_xlabel("ratio (compare models WITHIN each ratio)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=130)
    plt.close(fig)
    applied = {name: item["optimizer_applied"] for name, item in costs.items()}
    json_write(output / "comparison.json", {"rows": rows,
               "common_initialization_matches": len(set(init_hashes.values())) == 1,
               "epoch_index_order_matches": len({tuple(v) for v in order_hashes.values()}) == 1,
               "initial_hashes": init_hashes, "order_hashes": order_hashes,
               "applied_updates": applied, "equal_applied_updates": len(set(applied.values())) == 1,
               "scope": "A native x4 excluded from synthetic B/C/D deltas",
               "dedicated_total_train_seconds": sum(v["timing"]["train_seconds"] for k,v in costs.items() if k.startswith("B_")),
               "mixed_train_seconds": {k:v["timing"]["train_seconds"] for k,v in costs.items() if k.startswith(("C_", "D_"))}})

def archive_suite(output):
    output = Path(output)
    files = [p for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and not p.name.endswith((".tmp", ".lock")) and p.name != "suite_manifest.json"]
    manifest = {p.relative_to(output).as_posix(): sha_file(p) for p in files}
    json_write(output / "suite_manifest.json", {"hashes": manifest, "remote_storage_verified": False,
               "status": "local_package_created; FR validation and remote recovery not verified"})
    files.append(output / "suite_manifest.json")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(output.parent))
    # Verify every uncompressed ZIP member against the local manifest.
    import hashlib
    with zipfile.ZipFile(archive) as z:
        for rel, digest in manifest.items():
            if hashlib.sha256(z.read(output.name+"/"+rel)).hexdigest() != digest:
                raise ValueError("Archive member checksum mismatch")
    json_write(archive.with_suffix(".zip.sha256.json"), {"path": str(archive), "sha256": sha_file(archive),
               "members_verified": True, "remote_storage_verified": False})
    return archive
