import numpy as np
import torch

def iou_single(gt, dt):
    intersection = ((gt * dt) > 0).sum()
    union = ((gt + dt) > 0).sum()
    iou = intersection / (union + 1)
    return iou

def iou_batch(gt, dt, mean=True):
    intersection = ((gt * dt) > 0).sum(axis=(1,2))
    union = ((gt + dt) > 0).sum(axis=(1,2))
    iou = (intersection + 1) / (union + 1)
    if mean:
        return np.mean(iou)
    else:
        return iou

