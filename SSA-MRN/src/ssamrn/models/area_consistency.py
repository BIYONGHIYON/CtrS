"""Project an HR prediction onto the observed area-downsampled HSI."""

import torch
from torch.nn import functional as F


def project_area_consistency(prediction, lr_hsi, valid_mask=None, scale=4):
    """Match each complete scale×scale block mean to its LR measurement.

    Mixed valid/invalid blocks are unchanged so masked evaluation does not
    acquire a correction based on pixels outside the registration mask.
    The projection is appropriate only when LR was made by non-overlapping
    area averaging at the same scale.
    """
    if prediction.ndim != 4 or lr_hsi.ndim != 4:
        raise ValueError("Expected NCHW prediction and LR HSI")
    if scale < 1 or prediction.shape[:2] != lr_hsi.shape[:2] or (
        prediction.shape[-2] != lr_hsi.shape[-2] * scale
        or prediction.shape[-1] != lr_hsi.shape[-1] * scale
    ):
        raise ValueError("Prediction and LR HSI have incompatible shapes")
    if valid_mask is not None:
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask[:, 0]
        if valid_mask.shape != (prediction.shape[0], *prediction.shape[-2:]):
            raise ValueError("Valid mask must match the HR spatial shape")
        complete = F.avg_pool2d(valid_mask[:, None].float(), scale).eq(1)
    else:
        complete = None
    prediction = prediction.float()
    correction = lr_hsi.float() - F.avg_pool2d(prediction, scale)
    if complete is not None:
        correction = correction * complete
    return prediction + correction.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
