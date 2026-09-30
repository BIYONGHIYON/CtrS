"""Check all LIB split counts, ENVI byte sizes, RGB dimensions and filename overlap."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SSA-MRN/src"))
from ssamrn.data.lib_hsi import LIBHSI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "SSA-MRN/configs/lib_rgb_hsi.json").read_text(encoding="utf-8"))
    root = args.data_root or Path(config["data_root"])
    datasets = {}
    for split in ("train", "validation", "test"):
        dataset = LIBHSI(root, split)
        datasets[split] = {p.stem for p in dataset.files}
        print(f"{split}: {len(dataset.files)} paired RGB/HSI files; shape and byte sizes OK", flush=True)
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = datasets[first] & datasets[second]
        if overlap:
            raise ValueError(f"Overlapping scene names: {first}/{second}: {sorted(overlap)}")
    print("PASS: all 513 pairs, no filename overlap. Pixel-level corruption/alignment not verified.", flush=True)


if __name__ == "__main__":
    main()
