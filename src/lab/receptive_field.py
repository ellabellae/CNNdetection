"""Theoretical receptive field of a chain of convolution / pooling layers.

For a sequence of layers with kernel k, stride s, padding p (one value per
axis), the standard recurrence (Dumoulin & Visin; Araujo et al. 2019) gives
for each layer::

    jump_out  = jump_in * s          # input pixels between adjacent output cells
    rf_out    = rf_in + (k - 1) * jump_in
    start_out = start_in + ((k - 1) / 2 - p) * jump_in   # centre of the first output cell

A 1x1 convolution leaves rf unchanged. Upsampling by factor f divides the
jump by f and leaves rf unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RF:
    size: float  # receptive field size in input pixels
    jump: float  # stride of the output grid in input pixels
    start: float  # centre of output cell (0, 0) in input-pixel coordinates


def chain(layers: Iterable[tuple[int, int, int]], rf: RF = RF(1.0, 1.0, 0.0)) -> RF:
    """Fold (kernel, stride, padding) triples into a receptive field."""
    size, jump, start = rf.size, rf.jump, rf.start
    for k, s, p in layers:
        start = start + ((k - 1) / 2 - p) * jump
        size = size + (k - 1) * jump
        jump = jump * s
    return RF(size, jump, start)


def upsample(rf: RF, factor: float) -> RF:
    return RF(rf.size, rf.jump / factor, rf.start)


def cell_box(rf: RF, row: int, col: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Input-pixel bounding box (y0, y1, x0, x1), clipped, of output cell (row, col)."""
    cy = rf.start + row * rf.jump
    cx = rf.start + col * rf.jump
    half = rf.size / 2
    y0, y1 = int(max(0, round(cy - half))), int(min(height, round(cy + half)))
    x0, x1 = int(max(0, round(cx - half))), int(min(width, round(cx + half)))
    return y0, y1, x0, x1
