"""Resize an entire PanCollection sample without spatial cropping."""

import torch
import torch.nn.functional as F


def prepare_full_sample(sample, output_size):
    """Return PAN, LMS, MS batches at a requested 4:1 output scale."""
    pan, lms, ms = (sample[key] for key in ("pan", "lms", "ms"))
    if any(not isinstance(image, torch.Tensor) or image.ndim != 3 for image in (pan, lms, ms)):
        raise ValueError("Expected CHW tensors for pan, lms, and ms")
    height, width = pan.shape[-2:]
    if height != width or height % 4 or height < 4:
        raise ValueError("Expected a square PAN sample with sides divisible by 4")
    if pan.shape[0] != 1 or lms.shape[-2:] != (height, width):
        raise ValueError("PAN and LMS must cover the same full sample")
    if ms.shape != (lms.shape[0], height // 4, width // 4):
        raise ValueError("MS must cover the same sample at one-quarter resolution")
    if output_size < 4 or output_size % 4 or output_size > height:
        raise ValueError(f"size must be divisible by 4 and between 4 and {height}")

    def resize(image, side):
        batch = image.unsqueeze(0)
        if image.shape[-2:] == (side, side):
            return batch
        return F.interpolate(batch, size=(side, side), mode="area")

    return (
        resize(pan, output_size),
        resize(lms, output_size),
        resize(ms, output_size // 4),
    )
