"""Guard against silently swapping ENVI interleave/channel axes."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspect_pcb_pairs import band, open_cube


class TestEnviLayout(unittest.TestCase):
    def test_bip_bil_bsq_read_same_channels(self):
        expected = np.arange(2 * 3 * 4, dtype="<f4").reshape(2, 3, 4)
        layouts = {
            "bip": expected,
            "bil": expected.transpose(0, 2, 1),
            "bsq": expected.transpose(2, 0, 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            for layout, raw in layouts.items():
                with self.subTest(layout=layout):
                    path = Path(directory) / layout
                    raw.tofile(path)
                    header = {"lines": 2, "samples": 3, "bands": 4,
                              "interleave": layout, "data_type": 4,
                              "byte_order": 0, "header_offset": 0}
                    cube = open_cube(path, header)
                    for index in range(4):
                        np.testing.assert_array_equal(band(cube, layout, index), expected[:, :, index])


if __name__ == "__main__":
    unittest.main()
