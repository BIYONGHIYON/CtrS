"""Check that the model-output preview is saved and readable."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.preview import save_ms_rgb_preview, save_pan_preview, save_prediction


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

    def test_saves_grayscale_pan_input(self):
        pan = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            image_path = save_pan_preview(pan, directory, 16)
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.mode, "L")

    def test_quickbird_ms_rgb_band_order_and_display_size(self):
        ms = torch.zeros(1, 4, 4, 4)
        ms[0, 2, 0, 0] = 1  # Red is the third QuickBird band.
        ms[0, 1, 0, 1] = 1  # Green is the second.
        ms[0, 0, 1, 0] = 1  # Blue is the first.
        ms[0, 3, 3, 3] = 100  # NIR is excluded from RGB.
        with tempfile.TemporaryDirectory() as directory:
            image_path = save_ms_rgb_preview(ms, directory, 16)
            with Image.open(image_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(image.getpixel((4, 0)), (0, 255, 0))
                self.assertEqual(image.getpixel((0, 4)), (0, 0, 255))
                self.assertEqual(image.getpixel((12, 12)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
