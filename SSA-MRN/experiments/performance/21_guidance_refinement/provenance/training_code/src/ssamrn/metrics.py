"""PanCollection RR/FR metrics, following DLPan-Toolbox definitions.

FR PAN resizing uses scikit-image's antialiased bicubic approximation to
MATLAB imresize; D_s/QNR must be parity-checked before a strict paper claim.
"""

import numpy as np
from scipy.ndimage import convolve, convolve1d
from skimage.transform import resize


def _chw(image):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite CHW image")
    return array


def _onion(a, b):
    """Cayley-Dickson multiplication used by MATLAB onions_quality."""
    n = a.shape[-1]
    if n == 1:
        return a * b
    half = n // 2
    x, y = a[..., :half], a[..., half:]
    z, w = b[..., :half], b[..., half:]
    cy = np.concatenate((y[..., :1], -y[..., 1:]), axis=-1)
    cw = np.concatenate((w[..., :1], -w[..., 1:]), axis=-1)
    if n == 2:
        return np.concatenate((x * z - w * y, x * w + z * y), axis=-1)
    cx = np.concatenate((x[..., :1], -x[..., 1:]), axis=-1)
    return np.concatenate((_onion(x, z) - _onion(w, cy), _onion(cx, w) + _onion(z, y)), axis=-1)


def q2n(reference, fused, block=32):
    """Q4/Q8 with nonoverlapping 32-pixel blocks, matching q2n.m here."""
    reference, fused = _chw(reference), _chw(fused)
    if reference.shape != fused.shape:
        raise ValueError("Q2n shapes differ")
    channels, height, width = reference.shape
    if height % block or width % block or channels & (channels - 1):
        raise ValueError("Q2n requires block-aligned images and power-of-2 bands")
    def blocks(image):
        # MATLAB uint16 saturates/rounds values before the hypercomplex index.
        image = np.rint(np.clip(image, 0, 65535)).transpose(1, 2, 0)
        return image.reshape(height // block, block, width // block, block, channels).transpose(0, 2, 1, 3, 4).reshape(-1, block, block, channels)
    a, b = blocks(reference), blocks(fused)
    mean_a = a.mean(axis=(1, 2), keepdims=True)
    std_a = a.std(axis=(1, 2), ddof=1, keepdims=True)
    std_a = np.where(std_a == 0, np.finfo(float).eps, std_a)
    a = (a - mean_a) / std_a + 1
    b = (b - mean_a) / std_a + 1
    b[..., 1:] *= -1
    m1, m2 = a.mean(axis=(1, 2)), b.mean(axis=(1, 2))
    m1_sq, m2_sq = np.sum(m1 * m1, axis=-1), np.sum(m2 * m2, axis=-1)
    bias = np.divide(2 * np.sqrt(m1_sq * m2_sq), m1_sq + m2_sq,
                     out=np.zeros_like(m1_sq), where=(m1_sq + m2_sq) != 0)
    n = block * block
    variance_term = n / (n - 1) * (
        np.mean(np.sum(a * a + b * b, axis=-1), axis=(1, 2)) - m1_sq - m2_sq
    )
    quality = np.empty((len(a), channels), dtype=np.float64)
    nonzero = np.abs(variance_term) > 1e-12
    quality[~nonzero] = 0
    quality[~nonzero, -1] = bias[~nonzero]
    if np.any(nonzero):
        product_mean = _onion(a[nonzero], b[nonzero]).mean(axis=(1, 2))
        mean_product = _onion(m1[nonzero], m2[nonzero])
        quality[nonzero] = (product_mean - mean_product) * (n / (n - 1) * bias[nonzero] * 2 / variance_term[nonzero])[:, None]
    return float(np.linalg.norm(quality, axis=-1).mean())


def rr_metrics(reference, fused, *, ratio=4):
    """Return per-image RR metrics, not an image-concatenated score."""
    reference, fused = _chw(reference), _chw(fused)
    if reference.shape != fused.shape:
        raise ValueError("RR shapes differ")
    diff = fused - reference
    dot = np.sum(reference * fused, axis=0)
    norms = np.linalg.norm(reference, axis=0) * np.linalg.norm(fused, axis=0)
    valid = norms > 0
    sam = float(np.degrees(np.arccos(np.clip(dot[valid] / norms[valid], -1, 1))).mean())
    means = reference.mean(axis=(1, 2))
    band_mse = np.mean(diff * diff, axis=(1, 2))
    ergas = float(100 / ratio * np.sqrt(np.mean(band_mse / np.maximum(means * means, 1e-30))))
    # MATLAB SCC: Sobel gradient magnitudes on a 1-pixel cropped image.
    sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    def gradients(image):
        image = image[:, 1:-1, 1:-1]
        gy = np.stack([convolve(band, sobel, mode="constant") for band in image])
        gx = np.stack([convolve(band, sobel.T, mode="constant") for band in image])
        return np.hypot(gx, gy)
    a, b = gradients(reference), gradients(fused)
    scc = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    # The paper does not specify its PSNR convention. Use the empirically
    # closest documented choice: per-band PSNR at peak 2047, then average.
    psnr = float(np.mean(10 * np.log10(2047 * 2047 / np.maximum(band_mse, 1e-30))))
    return {"SAM": sam, "ERGAS": ergas, "PSNR": psnr,
            "SCC": scc, "Q2n": q2n(reference, fused)}


def _block_uqi(a, b, block=32):
    """Average MATLAB uqi over nonoverlapping blocks."""
    height, width = a.shape
    if a.shape != b.shape or height % block or width % block:
        raise ValueError("UQI shapes are not block aligned")
    def group(image):
        return image.reshape(height // block, block, width // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    x, y = group(a), group(b)
    mx, my = x.mean(axis=1), y.mean(axis=1)
    vx, vy = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
    cov = np.sum((x - mx[:, None]) * (y - my[:, None]), axis=1) / (x.shape[1] - 1)
    denominator = (vx + vy) * (mx * mx + my * my)
    quality = np.divide(4 * cov * mx * my, denominator,
                        out=np.zeros_like(mx), where=np.abs(denominator) > 1e-30)
    return float(quality.mean())


def _interp23tap(image, ratio=4):
    coeff = 2 * np.array([.5, .305334091185, 0, -.072698593239, 0, .021809577942,
                          0, -.005192756653, 0, .000807762146, 0, -.000060081482])
    coeff = np.r_[coeff[:0:-1], coeff]
    for stage in range(int(np.log2(ratio))):
        up = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
        offset = 1 if stage == 0 else 0
        up[offset::2, offset::2] = image
        image = convolve1d(convolve1d(up, coeff, axis=0, mode="wrap"), coeff, axis=1, mode="wrap")
    return image


def fr_metrics(fused, lms, ms, pan, *, ratio=4, block=32):
    """Toolbox-style QNR; PAN antialias resize is not MATLAB-bit-exact."""
    fused, lms, ms = _chw(fused), _chw(lms), _chw(ms)
    pan = np.asarray(pan, dtype=np.float64).squeeze()
    if fused.shape != lms.shape or pan.shape != fused.shape[1:] or ms.shape[1:] != (pan.shape[0] // ratio, pan.shape[1] // ratio):
        raise ValueError("FR input shapes differ")
    channels = fused.shape[0]
    spectral = [abs(_block_uqi(fused[i], fused[j], block) - _block_uqi(lms[i], lms[j], block))
                for i in range(channels) for j in range(i + 1, channels)]
    down = resize(pan, ms.shape[1:], order=3, mode="reflect", anti_aliasing=True, preserve_range=True)
    pan_filtered = _interp23tap(down, ratio)
    spatial = [abs(_block_uqi(fused[i], pan, block) - _block_uqi(lms[i], pan_filtered, block))
               for i in range(channels)]
    d_lambda, d_s = float(np.mean(spectral)), float(np.mean(spatial))
    return {"D_lambda": d_lambda, "D_s": d_s, "QNR": (1 - d_lambda) * (1 - d_s)}
