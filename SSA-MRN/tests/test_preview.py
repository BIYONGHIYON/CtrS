"""Check that the model-output preview is saved and readable."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.preview import save_prediction


class PreviewTests(unittest.TestCase):
    def test_saves_four_band_array_and_png(self):
        prediction = torch.arange(4 * 16 * 16, dtype=torch.float32).reshape(1, 4, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            array_path, image_path = save_prediction(prediction, directory, 16)
            self.assertEqual(np.load(array_path).shape, (4, 16, 16))
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
