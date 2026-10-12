"""Verify retained archive hashes. Does not retrain, infer, or recompute metrics."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Experiment name from experiments/catalog.json, or all")
    parser.add_argument("--artifact-root", type=Path, help="Check a newly generated report_manifest.json export")
    args = parser.parse_args()
    if args.artifact_root is not None:
        manifest = json.loads((args.artifact_root / "report_manifest.json").read_text())
        records = manifest.get("artifacts", manifest.get("files"))
        if not records:
            raise ValueError("No artifacts in the new export manifest")
        for name, expected in records.items():
            path = args.artifact_root / name
            if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError("Export mismatch: " + name)
        print(f"PASS: {len(records)} new export files; hashes only")
        return
    if not args.experiment:
        parser.error("Supply --experiment or --artifact-root")
    catalog = json.loads((ROOT / "experiments/catalog.json").read_text())
    inventory = json.loads((ROOT / "experiments/retained_files.json").read_text())["files"]
    if args.experiment != "all" and args.experiment not in catalog:
        parser.error("Unknown experiment: " + args.experiment)
    prefixes = None if args.experiment == "all" else catalog[args.experiment]["archives"]
    files = inventory if prefixes is None else {name: item for name, item in inventory.items() if any(name.startswith(prefix + "/") for prefix in prefixes)}
    if not files and args.experiment != "ms_observation":
        raise ValueError("No archived evidence listed")
    checkpoints = 0
    for name, expected in files.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Retained file mismatch: " + name)
        checkpoints += path.suffix == ".pt"
    if args.experiment == "ms_observation":
        profile = ROOT / catalog[args.experiment]["profile"]
        json.loads(profile.read_text())
    print(f"PASS: {len(files)} retained files, {checkpoints} checkpoints; hashes only, no model or metric re-evaluation")

if __name__ == "__main__":
    main()
