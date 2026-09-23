"""Check that smoke inputs downsample the entire aligned sample."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.resize import prepare_full_sample


class FullSampleResizeTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "pan": torch.zeros(1, 8, 8),
            "lms": torch.zeros(4, 8, 8),
            "ms": torch.zeros(4, 2, 2),
        }
        self.sample["pan"][0, -1, -1] = 8
        self.sample["lms"][:, -1, -1] = 8
        self.sample["ms"][:, -1, -1] = 4

    def test_full_sample_is_used_when_downsampling(self):
        pan, lms, ms = prepare_full_sample(self.sample, 4)
        self.assertEqual(pan.shape, (1, 1, 4, 4))
        self.assertEqual(lms.shape, (1, 4, 4, 4))
        self.assertEqual(ms.shape, (1, 4, 1, 1))
        self.assertEqual(pan[0, 0, -1, -1].item(), 2.0)
        self.assertEqual(lms[0, 0, -1, -1].item(), 2.0)
        self.assertEqual(ms[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(pan[0, 0, 0, 0].item(), 0.0)

    def test_original_size_keeps_full_sample(self):
        pan, lms, ms = prepare_full_sample(self.sample, 8)
        torch.testing.assert_close(pan[0], self.sample["pan"])
        torch.testing.assert_close(lms[0], self.sample["lms"])
        torch.testing.assert_close(ms[0], self.sample["ms"])

    def test_invalid_size_is_rejected(self):
        for size in (0, 6, 12):
            with self.subTest(size=size), self.assertRaises(ValueError):
                prepare_full_sample(self.sample, size)


if __name__ == "__main__":
    unittest.main()
