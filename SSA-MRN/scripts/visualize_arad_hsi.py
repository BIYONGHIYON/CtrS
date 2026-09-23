from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat


INPUT_DIR = Path("/Users/biyonghiyon_e/Downloads/데이터셋/NTIRE2020")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "arad_hsi_visualization"
FILES = [INPUT_DIR / f"ARAD_HS_{index:04d}.mat" for index in range(1, 4)]


def cie_1931_approx(wavelengths: np.ndarray) -> np.ndarray:
    """Approximate the CIE 1931 2-degree color matching functions."""
    wl = wavelengths.astype(np.float64)

    t1 = (wl - 442.0) * np.where(wl < 442.0, 0.0624, 0.0374)
    t2 = (wl - 599.8) * np.where(wl < 599.8, 0.0264, 0.0323)
    t3 = (wl - 501.1) * np.where(wl < 501.1, 0.0490, 0.0382)
    x_bar = (
        0.362 * np.exp(-0.5 * t1**2)
        + 1.056 * np.exp(-0.5 * t2**2)
        - 0.065 * np.exp(-0.5 * t3**2)
    )

    t1 = (wl - 568.8) * np.where(wl < 568.8, 0.0213, 0.0247)
    t2 = (wl - 530.9) * np.where(wl < 530.9, 0.0613, 0.0322)
    y_bar = 0.821 * np.exp(-0.5 * t1**2) + 0.286 * np.exp(-0.5 * t2**2)

    t1 = (wl - 437.0) * np.where(wl < 437.0, 0.0845, 0.0278)
    t2 = (wl - 459.0) * np.where(wl < 459.0, 0.0385, 0.0725)
    z_bar = 1.217 * np.exp(-0.5 * t1**2) + 0.681 * np.exp(-0.5 * t2**2)
    return np.stack([x_bar, y_bar, z_bar], axis=1).astype(np.float32)


def hyperspectral_to_srgb(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Convert a visible-range spectral cube to display-ready approximate sRGB."""
    cmf = cie_1931_approx(wavelengths)
    xyz = np.tensordot(cube, cmf, axes=([2], [0]))
    xyz /= max(float(cmf[:, 1].sum()), 1e-12)

    xyz_to_rgb = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float32,
    )
    rgb_linear = xyz @ xyz_to_rgb.T
    rgb_linear = np.clip(rgb_linear, 0.0, None)

    # Robust display exposure. This changes only visualization brightness.
    exposure = np.percentile(rgb_linear, 99.5)
    if exposure > 0:
        rgb_linear /= exposure
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)

    threshold = 0.0031308
    rgb = np.where(
        rgb_linear <= threshold,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(rgb, 0.0, 1.0)


def robust_band(band: np.ndarray) -> np.ndarray:
    low, high = np.percentile(band, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(band)
    return np.clip((band - low) / (high - low), 0.0, 1.0)


def save_scene(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = loadmat(path)
    cube = np.asarray(data["cube"], dtype=np.float32)
    wavelengths = np.asarray(data["bands"]).reshape(-1).astype(np.float32)
    rgb = hyperspectral_to_srgb(cube, wavelengths)

    rgb_uint8 = np.round(rgb * 255.0).astype(np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(OUTPUT_DIR / f"{path.stem}_rgb.png")

    selected_wavelengths = [420, 480, 540, 600, 660, 700]
    selected_indices = [int(np.argmin(np.abs(wavelengths - wl))) for wl in selected_wavelengths]

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("Approximate sRGB")
    ax_rgb.axis("off")

    band_axes = [fig.add_subplot(grid[row, col]) for row, col in [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]]
    for ax, index in zip(band_axes, selected_indices):
        ax.imshow(robust_band(cube[:, :, index]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Band {int(wavelengths[index])} nm")
        ax.axis("off")

    ax_spectrum = fig.add_subplot(grid[1, 3])
    mean_spectrum = cube.mean(axis=(0, 1))
    # A uniform spatial sample avoids a large temporary copy during percentiles.
    spectrum_sample = cube[::8, ::8, :].reshape(-1, cube.shape[2])
    p10 = np.percentile(spectrum_sample, 10, axis=0)
    p90 = np.percentile(spectrum_sample, 90, axis=0)
    ax_spectrum.plot(wavelengths, mean_spectrum, color="#2563eb", linewidth=2, label="Mean")
    ax_spectrum.fill_between(wavelengths, p10, p90, color="#93c5fd", alpha=0.45, label="10–90 percentile")
    ax_spectrum.set_title("Scene spectrum")
    ax_spectrum.set_xlabel("Wavelength (nm)")
    ax_spectrum.set_ylabel("Normalized intensity")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend(fontsize=8)

    fig.suptitle(f"{path.stem}: {cube.shape[0]}×{cube.shape[1]} pixels, {cube.shape[2]} bands", fontsize=16)
    fig.savefig(OUTPUT_DIR / f"{path.stem}_overview.png", dpi=180)
    plt.close(fig)

    stats = {
        "shape": tuple(int(x) for x in cube.shape),
        "wavelength_min_nm": int(wavelengths.min()),
        "wavelength_max_nm": int(wavelengths.max()),
        "value_min": float(cube.min()),
        "value_max": float(cube.max()),
        "value_mean": float(cube.mean()),
    }
    return rgb, wavelengths, stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb_images = []
    summaries = []
    for path in FILES:
        rgb, wavelengths, stats = save_scene(path)
        rgb_images.append((path.stem, rgb))
        summaries.append((path.stem, stats))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (name, rgb) in zip(axes, rgb_images):
        ax.imshow(rgb)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("ARAD hyperspectral cubes converted to approximate sRGB", fontsize=16)
    fig.savefig(OUTPUT_DIR / "ARAD_HS_0001-0003_rgb_comparison.png", dpi=180)
    plt.close(fig)

    summary_lines = [
        "ARAD HSI visualization summary",
        "RGB conversion: CIE 1931 approximation -> XYZ -> sRGB, with 99.5 percentile display exposure.",
        "The RGB files are visualization products, not original RGB-camera measurements.",
        "",
    ]
    for name, stats in summaries:
        summary_lines.append(f"{name}: {stats}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
