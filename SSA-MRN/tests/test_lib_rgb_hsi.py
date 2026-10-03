import sys
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssamrn.data.lib_hsi import LIBHSI, ScenePatchSampler, cube_view
from ssamrn.models.rgb_hsi import RGBHSISsaMRN
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet


class LIBTests(unittest.TestCase):
    def test_training_accumulation_resume_and_test_evaluation(self):
        script = Path(__file__).resolve().parents[1] / "scripts/train_lib.py"
        spec = importlib.util.spec_from_file_location("lib_trainer_test", script)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, split, **kwargs):
                self.files = [Path(split + "_scene.png")]
                generator = torch.Generator().manual_seed(10 if split == "train" else 20)
                self.rgb = torch.rand(3, 16, 16, generator=generator)
                self.gt = torch.rand(204, 16, 16, generator=generator)
                self.split = split
                self.per_scene = 5 if split == "train" else 1

            def __len__(self):
                return 5 if self.split == "train" else 1  # Last accumulation group has one batch.

            def __getitem__(self, index):
                from ssamrn.data.lib_hsi import downsample
                return {"rgb": self.rgb, "gt": self.gt, "lr_hsi": downsample(self.gt),
                        "scene": self.files[0].stem}

        with tempfile.TemporaryDirectory() as directory, patch.object(trainer, "LIBHSI", TinyDataset):
            output = Path(directory) / "run"
            config = {"data_root": str(Path(directory)/"data"), "ssai_dimension": 6,
                      "val_batch_size": 1, "patches_per_scene": 4, "learning_rate": 0.0001, "seed": 42}
            config.update(device="cpu", amp=False, latent_channels=4, patch_size=16, epochs=1,
                          workers=0, cpu_threads=2, batch_size=1, accumulation_steps=4,
                          output_dir=str(output))
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            arguments = [str(script), "--config", str(config_path)]
            with patch.object(sys, "argv", arguments):
                trainer.main()
            state = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(int(next(iter(state["optimizer"]["state"].values()))["step"]), 2)
            config["epochs"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(sys, "argv", arguments + ["--resume", str(output / "best.pt")]):
                trainer.main()
            resumed = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(int(next(iter(resumed["optimizer"]["state"].values()))["step"]), 4)
            with patch.object(sys, "argv", arguments + ["--evaluate"]):
                trainer.main()
            metrics = json.loads((output / "test/metrics.json").read_text())
            self.assertFalse(metrics["partial"])
            self.assertIn("test_scene", metrics["scenes"])

    def test_rotation_and_partial_file_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            hdr = Path(directory) / "sample.hdr"
            hdr.write_text("ENVI\nsamples = 4\nlines = 4\nbands = 2\ndata type = 4\nbyte order = 0\nheader offset = 0\ninterleave = BIL\n")
            cube = np.arange(32, dtype=np.float32).reshape(4, 4, 2)
            cube.transpose(0, 2, 1).tofile(hdr.with_suffix(".dat"))
            np.testing.assert_array_equal(cube_view(hdr), np.rot90(cube, 3))
            dataset = LIBHSI.__new__(LIBHSI)
            dataset.headers = [hdr]
            dataset._cache_scene, dataset._cache_cube = None, None
            with patch("ssamrn.data.lib_hsi.np.fromfile", wraps=np.fromfile) as read:
                np.testing.assert_array_equal(dataset._read_scene(0), np.rot90(cube, 3))
                dataset._read_scene(0)
                self.assertEqual(read.call_count, 1)
            dataset._stripe_scene, dataset._stripe_cache, dataset.stripe_rows = None, {}, 2
            for y, x, p in ((0, 0, 2), (1, 1, 2), (0, 0, 4)):
                expected = np.rot90(cube, 3)[y:y+p, x:x+p].transpose(2, 0, 1)
                np.testing.assert_array_equal(dataset._read_patch(0, y, x, p), expected)
            hdr.with_suffix(".dat").write_bytes(b"incomplete")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                cube_view(hdr)

    def test_scene_sampler_covers_all_patches_in_groups(self):
        dataset = LIBHSI.__new__(LIBHSI)
        dataset.files = [Path(f"scene_{i}") for i in range(3)]
        dataset.per_scene = 4
        indices = list(ScenePatchSampler(dataset))
        self.assertEqual(sorted(indices), list(range(12)))
        for offset in range(0, 12, 4):
            self.assertEqual(len({index // 4 for index in indices[offset:offset+4]}), 1)

    def test_rgb_all_bands_backward(self):
        torch.set_num_threads(2)
        model = RGBHSISsaMRN(204, 4, 6)
        rgb = torch.rand(1, 3, 16, 16, requires_grad=True)
        lr = torch.rand(1, 204, 4, 4, requires_grad=True)
        output = model(rgb, lr)
        self.assertEqual(output.shape, (1, 204, 16, 16))
        output.square().mean().backward()
        for gradient in (rgb.grad, lr.grad, model.encoder.weight.grad,
                         model.core.SSA_blocks[0].conv1t6.weight.grad, model.decoder.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_grouped_ssa_matches_branch_loop_forward_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        grouped = RGBHSISsaMRN(204, 4, 6).core
        loop = copy.deepcopy(grouped)
        loop.vectorized = False
        inputs = (torch.rand(2, 3, 16, 16), torch.rand(2, 4, 16, 16), torch.rand(2, 4, 4, 4))
        expected, actual = loop(*inputs), grouped(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for (name, first), (_, second) in zip(loop.named_parameters(), grouped.named_parameters()):
            if first.grad is None:
                self.assertIsNone(second.grad, name)
            else:
                torch.testing.assert_close(second.grad, first.grad, atol=1e-5, rtol=2e-4, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_degradation_matches_cpu_float32(self):
        from ssamrn.data.lib_hsi import downsample
        tensor = torch.rand(204, 32, 32)
        torch.testing.assert_close(downsample(tensor.cuda()).cpu(), downsample(tensor), atol=2e-6, rtol=2e-5)

    def test_pan_default_still_works(self):
        model = RestoredPansharpeningNet(4, 6)
        output = model(torch.rand(1, 1, 16, 16), torch.rand(1, 4, 16, 16), torch.rand(1, 4, 4, 4))
        self.assertEqual(output.shape, (1, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
