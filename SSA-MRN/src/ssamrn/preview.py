"""Save a model-output array and a display-only three-band PNG preview."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_prediction(output, output_dir, crop):
    """Save the C×H×W prediction and a contrast-stretched preview."""
    prediction = output.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if prediction.ndim != 3 or prediction.shape[0] < 3:
        raise ValueError("Expected a prediction with at least three bands")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"random_output_{crop}.npy"
    image_path = output_dir / f"random_output_preview_{crop}.png"
    np.save(array_path, prediction)

    # The first three bands are a visual check only, not calibrated RGB.
    first_three = np.moveaxis(prediction[:3], 0, -1)
    low, high = np.percentile(first_three, (1, 99), axis=(0, 1))
    scale = np.where(high > low, high - low, 1.0)
    preview = np.rint(np.clip((first_three - low) / scale, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(preview).save(image_path)
    return array_path, image_path
