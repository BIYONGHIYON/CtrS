"""Create a deterministic PCB-level split after checking all 53 pair filenames."""

import argparse
import json
from pathlib import Path

import numpy as np

from inspect_pcb_pairs import DEFAULT_DATASET, read_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("configs/pcb_split_seed42.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ids = set(range(1, 54))
    rgb_ids = {int(path.stem) for path in (args.dataset / "RGB").glob("[0-9]*.jpg")}
    hsi_ids = {int(path.name[3:]) for path in (args.dataset / "HSI").glob("pcb[0-9]*") if path.is_dir()}
    if rgb_ids != ids or hsi_ids != ids:
        raise ValueError(f"Incomplete pairs: missing RGB={sorted(ids-rgb_ids)}, HSI={sorted(ids-hsi_ids)}")
    for scene_id in ids:
        path = args.dataset / "HSI" / f"pcb{scene_id}" / f"pcb{scene_id}"
        header = read_header(path.with_suffix(".hdr"))
        if not path.is_file() or header["bands"] != 224:
            raise ValueError(f"Missing or non-224-band HSI: pcb{scene_id}")
    shuffled = np.random.default_rng(args.seed).permutation(sorted(ids))
    split = {"seed": args.seed, "unit": "PCB number, never patch", "dataset": "PCB-Vision version 1",
             "train": sorted(map(int, shuffled[:33])),
             "validation": sorted(map(int, shuffled[33:43])),
             "test": sorted(map(int, shuffled[43:]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(split[name]) for name in ("train", "validation", "test")}))


if __name__ == "__main__":
    main()
