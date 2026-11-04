import torch
import segmentation_models_pytorch as smp
from src.lab import models as M


def test_mammoclip_prefix_strip_and_key_match():
    enc = smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=3, classes=1).encoder
    fake = {"image_encoder." + k: torch.zeros_like(v) for k, v in enc.state_dict().items()}
    fake["text_encoder.something"] = torch.zeros(1)
    fake["logit_scale"] = torch.zeros(1)
    state = M.extract_mammoclip_encoder(fake)
    assert "logit_scale" not in state and all(not k.startswith("text_encoder") for k in state)
    missing, unexpected = M.load_encoder_state(enc, state)
    assert missing == [] and unexpected == []


def test_key_mismatch_is_reported():
    enc = smp.Unet("efficientnet-b0", encoder_weights=None, in_channels=3, classes=1).encoder
    state = {k: v for k, v in enc.state_dict().items() if "_blocks.0" not in k}
    state["extra.weight"] = torch.zeros(1)
    missing, unexpected = M.load_encoder_state(enc, state)
    assert any("_blocks.0" in k for k in missing) and unexpected == ["extra.weight"]


def test_lineup_entries_are_consistent():
    for spec in M.LINEUP.values():
        assert spec.input_convention in ("baseline", "mammoclip")
        assert (spec.in_channels == 3) == (spec.input_convention == "mammoclip")
        assert spec.native_preprocessing in ("raw", "clahe_median", "pca", "breast_crop")
