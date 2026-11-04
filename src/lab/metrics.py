"""Segmentation metrics on binary numpy masks."""
from __future__ import annotations

import numpy as np


def _bin(a: np.ndarray, threshold: float) -> np.ndarray:
    a = np.asarray(a)
    return a > threshold if a.dtype != bool else a


def dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """2|A∩B| / (|A|+|B|); 1.0 when both masks are empty."""
    p, t = _bin(pred, threshold), _bin(target, 0.5)
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, t).sum() / denom)


def iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """|A∩B| / |A∪B|; 1.0 when both masks are empty."""
    p, t = _bin(pred, threshold), _bin(target, 0.5)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def overlap(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> bool:
    """True if the predicted region touches the target at all (used to pick the saliency target)."""
    return bool(np.logical_and(_bin(pred, threshold), _bin(target, 0.5)).any())
