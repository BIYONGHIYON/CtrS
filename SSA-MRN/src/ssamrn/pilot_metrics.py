"""Spatial diagnostics with both endpoints of each finite difference masked."""
import torch


def gradient_error_stats(prediction, target, mask):
    error = prediction.float() - target.float()
    mx = mask[:, :, 1:] & mask[:, :, :-1]
    my = mask[:, 1:, :] & mask[:, :-1, :]
    ex = (error[:, :, :, 1:] - error[:, :, :, :-1]).square() * mx[:, None]
    ey = (error[:, :, 1:, :] - error[:, :, :-1, :]).square() * my[:, None]
    sse = ex.sum((1, 2, 3)) + ey.sum((1, 2, 3))
    count = (mx.sum((1, 2)) + my.sum((1, 2))) * target.shape[1]
    return sse, count
