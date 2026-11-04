"""End-to-end seam: the generator on a tiny config writes a valid, in-budget bundle."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.lab import build as B
from src.lab.bundle import BudgetExceeded


def _make_sample(tmp: Path, name="S1"):
    raw = tmp / "samples" / "raw"; raw.mkdir(parents=True)
    ann = tmp / "ann"; ann.mkdir()
    img = np.zeros((200, 160), np.uint8); img[30:180, 20:130] = 110; img[90:110, 60:80] = 230
    cv2.imwrite(str(raw / f"{name}.png"), img)
    poly = [{"x": 60, "y": 90}, {"x": 80, "y": 90}, {"x": 80, "y": 110}, {"x": 60, "y": 110}]
    (ann / f"{name}_AnnotationFile.json").write_text(json.dumps([{"label": "mass", "type": "Draw", "cgPoints": poly}]))
    return tmp / "samples", ann


def test_tiny_build_writes_valid_bundle(tmp_path):
    sample_dir, ann_dir = _make_sample(tmp_path)
    out = tmp_path / "out"
    args = B.parse_args(["--models", "random:efficientnet-b0:1", "--images", "S1", "--preprocessings", "raw", "breast_crop", "--always-preprocessings", "raw", "breast_crop",
                         "--resolutions", "64", "--untrained-resolutions", "64", "--analytic-resolutions", "64", "128", "--out", str(out),
                         "--sample-dir", str(sample_dir), "--annotations-dir", str(ann_dir), "--budget-mb", "5", "--tile", "16"])
    m = B.build(args, log=lambda *a: None)
    assert (out / "manifest.json").exists() and (out / "manifest.js").exists()
    assert set(m) >= {"meta", "models", "images", "runs"}
    model = m["models"]["random:efficientnet-b0:1"]
    assert model["totals"]["params"] > 0 and "64" in model["analytic"] and "128" in model["analytic"]
    assert model["analytic"]["128"]["macs"] > model["analytic"]["64"]["macs"]
    assert model["receptive_field"]["enc5"]["size"] > model["receptive_field"]["enc1"]["size"]
    assert len(m["runs"]) == 2    # raw + breast_crop at one resolution (breast_crop via --always-preprocessings)
    for r in m["runs"]:
        assert 0 <= r["dice"] <= 1 and r["gt_area_px"] > 0
        for f in list(r["files"].values()) + [s["sprite"] for s in r["stages"]] + list(r["saliency"].values()):
            assert (out / f).exists(), f
        stage_keys = [s["key"] for s in r["stages"]]
        assert stage_keys[0] == "enc0" and "dec4" in stage_keys
        assert r["stages"][-1]["shape"][1] == 64
        assert r["se"] and r["se"][0]["channels"] > 0
        assert "gt_encoder" in r["saliency"]
    img = m["images"]["S1"]
    assert img["mass"]["n_mass"] == 1 and (out / img["thumbs"]["raw"]).exists() and (out / img["gt"]["breast_crop"]).exists()
    assert img["meta"]["breast_crop"]["pixels_kept_fraction"] < 1


def test_over_budget_bundle_is_refused(tmp_path):
    sample_dir, ann_dir = _make_sample(tmp_path)
    args = B.parse_args(["--models", "random:efficientnet-b0:1", "--images", "S1", "--preprocessings", "raw", "--resolutions", "64", "--untrained-resolutions", "64",
                         "--analytic-resolutions", "64", "--out", str(tmp_path / "out"), "--sample-dir", str(sample_dir),
                         "--annotations-dir", str(ann_dir), "--budget-mb", "0.001"])
    with pytest.raises(BudgetExceeded):
        B.build(args, log=lambda *a: None)


def test_clahe_only_sample_runs_only_clahe(tmp_path):
    sample_dir = tmp_path / "samples"; (sample_dir / "clahe").mkdir(parents=True)
    ann = tmp_path / "ann"; ann.mkdir()
    img = np.full((120, 100), 90, np.uint8); img[40:60, 40:60] = 240
    cv2.imwrite(str(sample_dir / "clahe" / "S2_clahe.png"), img)
    (ann / "S2_AnnotationFile.json").write_text(json.dumps([{"label": "mass", "cgPoints": [{"x": 40, "y": 40}, {"x": 60, "y": 40}, {"x": 60, "y": 60}, {"x": 40, "y": 60}]}]))
    args = B.parse_args(["--models", "random:efficientnet-b0:1", "--images", "S2", "--resolutions", "64", "--untrained-resolutions", "64", "--analytic-resolutions", "64",
                         "--out", str(tmp_path / "out"), "--sample-dir", str(sample_dir), "--annotations-dir", str(ann)])
    m = B.build(args, log=lambda *a: None)
    assert [r["preprocessing"] for r in m["runs"]] == ["clahe_median"]
