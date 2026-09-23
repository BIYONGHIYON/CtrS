"""Print PanCollection H5 structure and the first sample's value range."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.pancollection import PanCollectionH5, inspect_h5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--sensor", default="QB")
    args = parser.parse_args()

    for key, metadata in inspect_h5(args.path).items():
        print(f"{key}: {metadata}")
    dataset = PanCollectionH5(args.path, sensor=args.sensor)
    print(f"samples: {len(dataset)}, channels: {dataset.channels}")
    for key, value in dataset[0].items():
        print(f"{key}: normalized min={value.min().item():.5f}, max={value.max().item():.5f}")


if __name__ == "__main__":
    main()
