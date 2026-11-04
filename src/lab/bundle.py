"""Write the lab's data bundle: feature-map sprites, overlays, and the size budget."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


class BudgetExceeded(RuntimeError):
    pass


def dir_bytes(path: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def check_budget(path: str | Path, budget_mb: float) -> int:
    n = dir_bytes(path)
    if n > budget_mb * 1e6:
        raise BudgetExceeded(f"bundle at {path} is {n / 1e6:.1f} MB, over the {budget_mb} MB budget")
    return n


def write_png(path: str | Path, arr: np.ndarray) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr, PNG_PARAMS)
    if not ok:
        raise IOError(f"could not write {path}")
    return Path(path).stat().st_size


def normalize_u8(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robust per-array scaling to uint8."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _resize(a: np.ndarray, tile: int) -> np.ndarray:
    interp = cv2.INTER_NEAREST if a.shape[0] <= tile else cv2.INTER_AREA
    return cv2.resize(a, (tile, tile), interpolation=interp)


def feature_sprite(feat: np.ndarray, k: int = 16, tile: int = 96, cols: int = 8) -> tuple[np.ndarray, dict]:
    """Tile the ``k`` highest-variance channels of a (C, H, W) map into one uint8 sprite.

    Returns (sprite, meta) where meta lists the chosen channel indices, per-channel
    statistics, and the grid geometry the page needs to cut the sprite.
    """
    c = feat.shape[0]
    var = feat.reshape(c, -1).var(axis=1)
    idx = np.argsort(-var)[: min(k, c)]
    rows = int(np.ceil(len(idx) / cols))
    sprite = np.zeros((rows * tile, cols * tile), np.uint8)
    stats = []
    for n, ch in enumerate(idx):
        m = feat[ch]
        r, cc = divmod(n, cols)
        sprite[r * tile:(r + 1) * tile, cc * tile:(cc + 1) * tile] = _resize(normalize_u8(m), tile)
        stats.append({"channel": int(ch), "mean": float(m.mean()), "std": float(m.std()), "sparsity": float((m <= 0).mean())})
    meta = {"tile": tile, "cols": cols, "rows": rows, "channels_total": int(c), "map_shape": [int(feat.shape[1]), int(feat.shape[2])], "channels": stats}
    return sprite, meta


def gate_strip(gate: np.ndarray, height: int = 8) -> np.ndarray:
    """A (height, C) uint8 strip of squeeze-excite gate values (0..1 -> 0..255)."""
    row = np.clip(gate * 255, 0, 255).astype(np.uint8)[None, :]
    return np.repeat(row, height, axis=0)


def thumb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def heat_u8(cam: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(np.clip(cam * 255, 0, 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_LINEAR)


def mask_u8(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize((mask > 0).astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST)
