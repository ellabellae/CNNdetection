import numpy as np
from src.lab import preprocess as P


def _breast_image(h=200, w=160):
    img = np.zeros((h, w), np.uint8)
    img[40:170, 20:120] = 120          # "breast"
    img[90:100, 60:70] = 220           # "mass"
    return img


def test_breast_crop_bounds_contain_tissue_and_mask_follows():
    img = _breast_image()
    mask = np.zeros_like(img); mask[90:100, 60:70] = 1
    out = P.apply("breast_crop", img, mask)
    y0, y1, x0, x1 = out.meta["bbox"]
    assert y0 <= 40 and y1 >= 170 and x0 <= 20 and x1 >= 120
    assert out.image.shape == out.mask.shape
    assert out.mask.sum() == mask.sum()           # nothing of the mass was cut off
    assert 0 < out.meta["pixels_kept_fraction"] < 1


def test_each_preprocessing_returns_uint8_same_shape_except_crop():
    img = _breast_image()
    for name in ("raw", "clahe_median", "pca"):
        out = P.apply(name, img)
        assert out.image.dtype == np.uint8 and out.image.shape == img.shape, name


def test_input_conventions():
    img = _breast_image()
    base = P.to_input(img, 64, "baseline")
    assert base.shape == (1, 64, 64) and base.max() <= 1.0 + 1e-6 and base.min() >= 0
    mc = P.to_input(img, 64, "mammoclip")
    assert mc.shape == (3, 64, 64) and np.allclose(mc[0], mc[2])
    # min-max to [0,1] then standardise: the max pixel maps to (1 - mean) / std
    assert abs(mc.max() - (1 - P.MAMMOCLIP_MEAN) / P.MAMMOCLIP_STD) < 1e-4


def test_input_bytes_scale_with_resolution():
    assert P.input_bytes(256)["uint8"] == 65536
    assert P.input_bytes(512)["float32"] == 4 * 512 * 512
    assert P.input_bytes(1024, 3)["uint8"] == 3 * 1024 * 1024
