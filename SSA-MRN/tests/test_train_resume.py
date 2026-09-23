"""Verify one synthetic training step and checkpoint resumption."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


class TrainResumeTests(unittest.TestCase):
    def test_train_then_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for offset, name in enumerate(("train", "val")):
                with h5py.File(root / f"{name}.h5", "w") as file:
                    for key, shape in {
                        "pan": (1, 1, 16, 16),
                        "lms": (1, 4, 16, 16),
                        "ms": (1, 4, 4, 4),
                        "gt": (1, 4, 16, 16),
                    }.items():
                        file[key] = np.random.default_rng(len(key) + offset).uniform(0, 2047, shape)

            checkpoint = root / "checkpoints/latest.pt"
            command = [
                sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/train.py"),
                "--train", str(root / "train.h5"),
                "--val", str(root / "val.h5"),
                "--batch-size", "1",
                "--checkpoint-dir", str(checkpoint.parent),
            ]
            first = subprocess.run(command + ["--epochs", "1"], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(checkpoint.is_file())
            second = subprocess.run(command + ["--epochs", "2", "--resume", str(checkpoint)],
                                    capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(torch.load(checkpoint, weights_only=True)["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
