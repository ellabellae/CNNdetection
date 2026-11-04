"""Grad-CAM for a segmentation U-Net.

Grad-CAM explains one scalar. For segmentation we use the mean logit inside a
region (the ground-truth mass, or the model's own predicted region when it
missed the mass entirely). The class-activation map at a chosen stage is the
ReLU of the gradient-weighted sum of that stage's channels, upsampled to the
input size and scaled to [0, 1].
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def _stage_module(model: nn.Module, stage: str) -> nn.Module:
    if stage == "encoder_last":
        return model.encoder._blocks[-1]
    if stage.startswith("dec"):
        return model.decoder.blocks[int(stage[3:])]
    raise ValueError(f"unknown saliency stage {stage!r}")


def grad_cam(model: nn.Module, x: torch.Tensor, region: np.ndarray, stage: str = "encoder_last") -> np.ndarray:
    """Return a (H, W) float32 heat map in [0, 1] for the mean logit inside ``region`` (H, W binary)."""
    acts: dict[str, torch.Tensor] = {}
    mod = _stage_module(model, stage)

    def fwd(m, inp, out):
        out.retain_grad()
        acts["a"] = out

    h = mod.register_forward_hook(fwd)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(x)
        r = torch.from_numpy(region.astype(np.float32)).to(logits.device)[None, None]
        if r.shape[-2:] != logits.shape[-2:]:
            r = torch.nn.functional.interpolate(r, size=logits.shape[-2:], mode="nearest")
        denom = r.sum()
        target = (logits * r).sum() / denom if denom > 0 else logits.mean()
        target.backward()
    h.remove()
    a = acts["a"]
    g = a.grad
    if g is None:
        return np.zeros(x.shape[-2:], np.float32)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(dim=1, keepdim=True))[0, 0].detach().cpu().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
    if cam.max() > 0:
        cam = cam / cam.max()
    model.zero_grad(set_to_none=True)
    return cam.astype(np.float32)
