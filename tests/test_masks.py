import numpy as np
from src.lab import masks


def _sq(x0, y0, x1, y1, label):
    return {"label": label, "type": "Draw", "cgPoints": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]}


def test_mass_only_excludes_other_findings():
    ann = [_sq(2, 2, 6, 6, "mass"), _sq(10, 10, 14, 14, "Calcification")]
    m = masks.rasterize(ann, 20, 20)
    assert m.dtype == np.uint8
    assert m[4, 4] == 1 and m[12, 12] == 0
    everything = masks.rasterize(ann, 20, 20, labels=None)
    assert everything[12, 12] == 1


def test_degenerate_polygon_is_skipped():
    ann = [{"label": "mass", "cgPoints": [{"x": 1, "y": 1}, {"x": 2, "y": 2}]}]
    assert masks.rasterize(ann, 5, 5).sum() == 0


def test_summary_counts_and_area():
    ann = [_sq(0, 0, 10, 10, "mass"), _sq(0, 0, 2, 2, "lipoma")]
    s = masks.mass_summary(ann)
    assert s["n_shapes"] == 2 and s["n_mass"] == 1
    assert abs(s["mass_area_px"] - 100.0) < 1e-6
    assert s["labels"] == ["mass", "lipoma"]
