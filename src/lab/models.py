"""Model registry for the lab: build any lineup entry by name.

Each entry is a U-Net from ``segmentation_models_pytorch``. The baseline
entries load the team's trained checkpoints; the Mammo-CLIP entries load a
mammography-pretrained encoder (extracted once from the published CLIP
checkpoint) and leave the decoder random, which the manifest flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

MAMMOCLIP_PREFIX = "image_encoder."


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    encoder: str
    in_channels: int
    input_convention: str  # "baseline" | "mammoclip"
    weights: str  # "checkpoint" | "mammoclip-encoder" | "imagenet" | "random"
    weights_path: str | None
    native_preprocessing: str
    decoder_trained: bool
    source: str
    license: str


LINEUP: dict[str, ModelSpec] = {
    "unet_b0_raw": ModelSpec(
        "unet_b0_raw", "U-Net · EfficientNet-B0 · trained on raw", "efficientnet-b0", 1, "baseline",
        "checkpoint", "data/weights/model_raw.pth", "raw", True,
        "Team checkpoint (BME 303 baseline, 20 epochs, BCE)", "project",
    ),
    "unet_b0_clahe": ModelSpec(
        "unet_b0_clahe", "U-Net · EfficientNet-B0 · trained on CLAHE+median", "efficientnet-b0", 1, "baseline",
        "checkpoint", "data/weights/model_clahe.pth", "clahe_median", True,
        "Team checkpoint (BME 303 baseline, 20 epochs, BCE)", "project",
    ),
    "unet_b0_imagenet": ModelSpec(
        "unet_b0_imagenet", "U-Net · EfficientNet-B0 · ImageNet encoder, untrained decoder", "efficientnet-b0", 1, "baseline",
        "imagenet", None, "raw", False,
        "segmentation_models_pytorch ImageNet weights", "Apache-2.0 / ImageNet terms",
    ),
    "unet_mammoclip_b2": ModelSpec(
        "unet_mammoclip_b2", "U-Net · Mammo-CLIP EfficientNet-B2 encoder, untrained decoder", "efficientnet-b2", 3, "mammoclip",
        "mammoclip-encoder", "data/weights/mammoclip_b2_encoder.pth", "raw", False,
        "Ghosh et al. 2024, Mammo-CLIP (arXiv 2405.12255), HF shawn24/Mammo-CLIP", "CC BY-NC-SA 4.0",
    ),
    "unet_mammoclip_b5": ModelSpec(
        "unet_mammoclip_b5", "U-Net · Mammo-CLIP EfficientNet-B5 encoder, untrained decoder", "efficientnet-b5", 3, "mammoclip",
        "mammoclip-encoder", "data/weights/mammoclip_b5_encoder.pth", "raw", False,
        "Ghosh et al. 2024, Mammo-CLIP (arXiv 2405.12255), HF shawn24/Mammo-CLIP", "CC BY-NC-SA 4.0",
    ),
}


def extract_mammoclip_encoder(state: dict) -> dict:
    """Strip the ``image_encoder.`` prefix from a Mammo-CLIP ``model`` state dict."""
    return {k[len(MAMMOCLIP_PREFIX):]: v for k, v in state.items() if k.startswith(MAMMOCLIP_PREFIX)}


def load_encoder_state(encoder: nn.Module, state: dict) -> tuple[list[str], list[str]]:
    """Load into an smp encoder and return (missing, unexpected) keys.

    smp's EfficientNet encoder overrides ``load_state_dict`` and returns None,
    so we compare key sets ourselves and call the base implementation.
    """
    own = set(encoder.state_dict().keys())
    given = set(state.keys())
    missing = sorted(own - given)
    unexpected = sorted(given - own)
    nn.Module.load_state_dict(encoder, {k: v for k, v in state.items() if k in own}, strict=False)
    return missing, unexpected


def build(spec: ModelSpec, device: str = "cpu", require_weights: bool = True) -> nn.Module:
    """Instantiate a lineup entry. Raises FileNotFoundError if its weights are required and absent."""
    if spec.weights == "imagenet":
        model = smp.Unet(spec.encoder, encoder_weights="imagenet", in_channels=spec.in_channels, classes=1)
    elif spec.weights == "random":
        model = smp.Unet(spec.encoder, encoder_weights=None, in_channels=spec.in_channels, classes=1)
    else:
        path = Path(spec.weights_path or "")
        if not path.exists():
            if require_weights:
                raise FileNotFoundError(f"{spec.key}: weights not found at {path}")
            model = smp.Unet(spec.encoder, encoder_weights=None, in_channels=spec.in_channels, classes=1)
        else:
            model = smp.Unet(spec.encoder, encoder_weights=None, in_channels=spec.in_channels, classes=1)
            state = torch.load(path, map_location=device, weights_only=True)
            if spec.weights == "checkpoint":
                model.load_state_dict(state)
            else:  # mammoclip-encoder: encoder only, decoder stays random
                torch.manual_seed(0)
                missing, unexpected = load_encoder_state(model.encoder, state)
                if missing or unexpected:
                    raise RuntimeError(f"{spec.key}: encoder keys mismatch; missing={missing[:3]} unexpected={unexpected[:3]}")
    return model.to(device).eval()
