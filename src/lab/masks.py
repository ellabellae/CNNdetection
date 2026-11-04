"""Ground-truth masks from TOMPEI-CMMD annotation JSON.

Each annotation file is a list of shapes. Every shape has ``cgPoints`` (polygon
vertices as ``{"x": .., "y": ..}`` dicts), a ``label`` naming the finding
(``mass``, ``calcification``, ...) and a ``type`` that is always ``"Draw"``.
The baseline notebook rasterised every shape; the lab rasterises only the
labels asked for (masses by default), per the spec.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage.draw import polygon as sk_polygon

MASS_LABELS: tuple[str, ...] = ("mass",)


def iter_shapes(ann: object) -> list[dict]:
    """Return the list of shape dicts from a parsed annotation (list, or dict wrapper)."""
    if isinstance(ann, list):
        return [s for s in ann if isinstance(s, dict)]
    if isinstance(ann, dict):
        for key in ("annotations", "shapes"):
            if isinstance(ann.get(key), list):
                return [s for s in ann[key] if isinstance(s, dict)]
        return [ann]
    return []


def shape_label(shape: dict) -> str:
    return str(shape.get("label", "")).strip().lower()


def finding_labels(ann: object) -> list[str]:
    """All finding labels in an annotation, in file order."""
    return [shape_label(s) for s in iter_shapes(ann)]


def polygon_points(shape: dict) -> np.ndarray | None:
    """Polygon vertices as an (N, 2) float array of [x, y], or None if fewer than 3."""
    pts = shape.get("cgPoints", [])
    out = []
    for p in pts:
        if isinstance(p, dict):
            if "x" in p and "y" in p:
                out.append((float(p["x"]), float(p["y"])))
            elif "X" in p and "Y" in p:
                out.append((float(p["X"]), float(p["Y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=np.float64)


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area in pixels of an (N, 2) polygon."""
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def rasterize(ann: object, height: int, width: int, labels: Sequence[str] | None = MASS_LABELS) -> np.ndarray:
    """Binary uint8 mask (H, W) of the shapes whose label is in ``labels``.

    ``labels=None`` reproduces the baseline notebook: every shape is drawn.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    wanted = None if labels is None else {l.lower() for l in labels}
    for shape in iter_shapes(ann):
        if wanted is not None and shape_label(shape) not in wanted:
            continue
        pts = polygon_points(shape)
        if pts is None:
            continue
        rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(height, width))
        mask[rr, cc] = 1
    return mask


def load_annotation(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def load_mask(path: str | Path, height: int, width: int, labels: Sequence[str] | None = MASS_LABELS) -> np.ndarray:
    return rasterize(load_annotation(path), height, width, labels)


def mass_summary(ann: object) -> dict:
    """Counts and total polygon area of masses, plus the other labels present."""
    shapes = iter_shapes(ann)
    masses = [polygon_points(s) for s in shapes if shape_label(s) in MASS_LABELS]
    masses = [m for m in masses if m is not None]
    return {
        "n_shapes": len(shapes),
        "n_mass": len(masses),
        "mass_area_px": sum(polygon_area(m) for m in masses),
        "labels": finding_labels(ann),
    }
