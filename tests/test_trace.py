import numpy as np
import torch
import segmentation_models_pytorch as smp
from src.lab import trace as T
from src.lab.saliency import grad_cam


def _model():
    torch.manual_seed(0)
    return smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=1, classes=1).eval()


def test_macs_match_hand_computation_for_stem_and_block1_depthwise():
    m = _model()
    t = T.trace(m, torch.rand(1, 1, 64, 64), capture=False)
    by = {l.name: l for l in t.layers}
    stem = by["encoder._conv_stem"]
    # 32 out channels at 32x32 output, 1 input channel, 3x3 kernel
    assert stem.macs == 32 * 32 * 32 * 1 * 9
    dw = by["encoder._blocks.1._depthwise_conv"]
    assert dw.depthwise and dw.macs == 96 * 16 * 16 * 1 * 9   # groups == channels -> 1 input channel per group
    assert stem.out_bytes == 4 * 32 * 32 * 32


def test_stage_capture_shapes_and_se_gates():
    m = _model()
    t = T.trace(m, torch.rand(1, 1, 64, 64))
    assert t.stage_tensors["enc5"].shape == (320, 2, 2)
    assert t.stage_tensors["dec4"].shape == (16, 64, 64)
    assert t.stage_tensors["head"].shape == (1, 64, 64) and t.logits.shape == (64, 64)
    gate = t.se_gates["encoder._blocks.1"]
    assert gate.shape == (96,) and gate.min() >= 0 and gate.max() <= 1
    stages = T.stage_list(t)
    assert [s["key"] for s in stages][:2] == ["enc0", "enc1"] and stages[-1]["key"] == "head"
    assert t.decoder_blocks[0]["skip_shape"] == [1, 112, 4, 4]
    assert T.encoder_groups_by_level(t)["enc5"][0].startswith("encoder._blocks")


def test_totals_and_timing():
    m = _model()
    t = T.trace(m, torch.rand(1, 1, 64, 64), capture=False)
    tot = t.totals()
    assert tot["macs"] > 0 and tot["activation_bytes_total"] >= tot["activation_bytes_largest"] > 0
    assert tot["params_traced"] + tot["params_unused"] == tot["params"] == sum(p.numel() for p in m.parameters())
    assert tot["params_unused"] == 412160   # EfficientNet-B0 _conv_head (320x1280) + its BatchNorm, never executed by smp
    tm = T.time_forward(m, torch.rand(1, 1, 64, 64), warmup=1, runs=2)
    assert tm["median_s"] > 0
    st = T.time_stages(m, torch.rand(1, 1, 64, 64), runs=1)
    assert "encoder" in st and "dec0" in st and "head" in st


def test_grad_cam_is_in_range_and_input_sized():
    m = _model()
    x = torch.rand(1, 1, 64, 64)
    region = np.zeros((64, 64), np.uint8); region[20:40, 20:40] = 1
    cam = grad_cam(m, x, region, "encoder_last")
    assert cam.shape == (64, 64) and cam.min() >= 0 and cam.max() <= 1.0 + 1e-6
    cam2 = grad_cam(m, x, region, "dec2")
    assert cam2.shape == (64, 64)
