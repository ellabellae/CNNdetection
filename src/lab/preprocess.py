"""Preprocessing axis of the lab: raw, CLAHE + median, PCA, breast crop.

All functions take and return uint8 (H, W) grayscale arrays. ``apply`` also
carries the ground-truth mask through any geometric change (the crop) so
image and mask stay aligned. The first three reproduce the baseline notebook
exactly; ``breast_crop`` is the roadmap-step-2 preview.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA

PREPROCESSING_NAMES = ("raw", "clahe_median", "pca", "breast_crop")


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def raw(img: np.ndarray) -> np.ndarray:
    return img.copy()


def clahe_median(img: np.ndarray, median_size: int = 3, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Baseline recipe: min-max normalise, median filter, CLAHE."""
    x = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x = median_filter(x, size=median_size).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(x)


def pca_filter(img: np.ndarray, variance: float = 0.9) -> np.ndarray:
    """Baseline PCA 'filter': treat rows as samples, keep ``variance`` of column variance, reconstruct."""
    x = img.astype(np.float32)
    pca = PCA(n_components=variance)
    rec = pca.inverse_transform(pca.fit_transform(x))
    rec = rec - rec.min()
    if rec.max() > 0:
        rec = rec / rec.max() * 255
    return _u8(rec)


def breast_bbox(img: np.ndarray, margin: int = 16) -> tuple[int, int, int, int]:
    """Bounding box (y0, y1, x0, x1) of the breast: Otsu threshold, largest connected component."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 0, img.shape[0], 0, img.shape[1]
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP], stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
    y0, y1 = max(0, y - margin), min(img.shape[0], y + h + margin)
    x0, x1 = max(0, x - margin), min(img.shape[1], x + w + margin)
    return int(y0), int(y1), int(x0), int(x1)


def breast_crop(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    y0, y1, x0, x1 = breast_bbox(img)
    return img[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


@dataclass
class Preprocessed:
    name: str
    image: np.ndarray
    mask: np.ndarray | None
    meta: dict = field(default_factory=dict)


def apply(name: str, img: np.ndarray, mask: np.ndarray | None = None) -> Preprocessed:
    """Apply one named preprocessing to an image (and carry the mask through a crop)."""
    if name == "raw":
        return Preprocessed(name, raw(img), mask)
    if name == "clahe_median":
        return Preprocessed(name, clahe_median(img), mask)
    if name == "pca":
        return Preprocessed(name, pca_filter(img), mask)
    if name == "breast_crop":
        out, (y0, y1, x0, x1) = breast_crop(img)
        m = mask[y0:y1, x0:x1].copy() if mask is not None else None
        kept = out.size / img.size
        return Preprocessed(name, out, m, {"bbox": [y0, y1, x0, x1], "pixels_kept_fraction": kept})
    raise ValueError(f"unknown preprocessing {name!r}; choose from {PREPROCESSING_NAMES}")


# --------------------------------------------------------------------------- #
# Model input conventions
# --------------------------------------------------------------------------- #
MAMMOCLIP_MEAN = 0.3089279
MAMMOCLIP_STD = 0.25053555408335154


def to_input(img: np.ndarray, size: int, convention: str = "baseline") -> np.ndarray:
    """Resize and normalise a uint8 image into a float32 (C, size, size) array.

    ``baseline``: 1 channel, divide by the image max (the notebook's dataset).
    ``mammoclip``: 3 identical channels, per-image min-max to [0, 1], then
    ``(x - mean) / std`` with Mammo-CLIP's constants.
    """
    x = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    if convention == "baseline":
        x = x / (x.max() + 1e-8)
        return x[None]
    if convention == "mammoclip":
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        x = (x - MAMMOCLIP_MEAN) / MAMMOCLIP_STD
        return np.repeat(x[None], 3, axis=0)
    raise ValueError(f"unknown input convention {convention!r}")


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def input_bytes(size: int, channels: int = 1) -> dict[str, int]:
    """Bytes one input image occupies at this resolution, for the dataset-packaging question."""
    n = size * size * channels
    return {"uint8": n, "float16": 2 * n, "float32": 4 * n}
