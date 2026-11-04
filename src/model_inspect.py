"""Inspect the segmentation network: layer table, kernels, feature maps, architecture JSON.

The baseline model is a U-Net whose encoder is EfficientNet-B0
(``segmentation_models_pytorch``). This script walks every leaf module with
forward hooks and records what each one does to the tensor (kernel size,
stride, channels, input/output shape, parameter count). The result feeds the
interactive explainer in ``docs/explainer/`` and gives a human-readable
``torchinfo`` summary.

Examples
--------
Architecture only (downloads ImageNet encoder weights the first time)::

    python -m src.model_inspect --out docs/explainer/architecture.json

With a trained checkpoint and a real mammogram, also saving kernel and
feature-map figures::

    python -m src.model_inspect --weights model_clahe.pth --image some.png \
        --out docs/explainer/architecture.json --figdir docs/figures/model
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:  # optional, only for --figdir
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(
    encoder: str = "efficientnet-b0",
    in_channels: int = 1,
    encoder_weights: str | None = "imagenet",
    weights_path: str | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Build the baseline U-Net exactly as the notebook does, optionally loading a checkpoint."""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None if weights_path else encoder_weights,
        in_channels=in_channels,
        classes=1,
    )
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Describing leaf modules
# --------------------------------------------------------------------------- #
def _tolist(v: Any) -> Any:
    if isinstance(v, (tuple, list)):
        return [int(x) for x in v]
    return v


def describe_module(m: nn.Module) -> dict[str, Any]:
    """Return the hyperparameters that define what a leaf module computes."""
    d: dict[str, Any] = {"type": type(m).__name__}
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        d.update(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=_tolist(m.kernel_size),
            stride=_tolist(m.stride),
            padding=_tolist(m.padding),
            groups=m.groups,
            bias=m.bias is not None,
        )
        # Depthwise conv: one filter per input channel (groups == in_channels).
        d["depthwise"] = m.groups > 1 and m.groups == m.in_channels
        # Pointwise conv: 1x1 kernel that only mixes channels.
        d["pointwise"] = list(m.kernel_size) == [1, 1] if isinstance(m.kernel_size, tuple) else m.kernel_size == 1
    elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
        d.update(kernel_size=_tolist(m.kernel_size), stride=_tolist(m.stride), padding=_tolist(m.padding))
    elif isinstance(m, nn.AdaptiveAvgPool2d):
        d.update(output_size=_tolist(m.output_size))
    elif isinstance(m, nn.Upsample):
        d.update(scale_factor=m.scale_factor, mode=m.mode)
    elif isinstance(m, nn.BatchNorm2d):
        d.update(num_features=m.num_features, eps=m.eps, momentum=m.momentum)
    elif isinstance(m, nn.Linear):
        d.update(in_features=m.in_features, out_features=m.out_features)
    elif isinstance(m, nn.Dropout):
        d.update(p=m.p)
    return d


def _shape(t: Any) -> list[int] | None:
    if isinstance(t, torch.Tensor):
        return list(t.shape)
    if isinstance(t, (tuple, list)) and t and isinstance(t[0], torch.Tensor):
        return list(t[0].shape)
    return None


def _group_of(name: str) -> str:
    """Map a module path to a coarse block name used by the explainer."""
    parts = name.split(".")
    if parts[0] == "encoder":
        if len(parts) >= 3 and parts[1] == "_blocks":
            return f"encoder._blocks.{parts[2]}"
        return f"encoder.{parts[1]}" if len(parts) > 1 else "encoder"
    if parts[0] == "decoder":
        if len(parts) >= 3 and parts[1] == "blocks":
            return f"decoder.blocks.{parts[2]}"
        return "decoder"
    return parts[0]


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
def trace_model(model: nn.Module, x: torch.Tensor) -> dict[str, Any]:
    """Run one forward pass and record every leaf module, encoder level and decoder block."""
    layers: list[dict[str, Any]] = []
    encoder_levels: list[dict[str, Any]] = []
    decoder_blocks: list[dict[str, Any]] = []
    handles = []
    order = {"i": 0}

    def leaf_hook(name: str):
        def hook(m: nn.Module, inp: tuple, out: Any) -> None:
            order["i"] += 1
            rec = describe_module(m)
            rec.update(
                order=order["i"],
                name=name,
                group=_group_of(name),
                in_shape=_shape(inp),
                out_shape=_shape(out),
                params=sum(p.numel() for p in m.parameters(recurse=False)),
            )
            layers.append(rec)

        return hook

    # A Conv2d subclass (e.g. EfficientNet's Conv2dStaticSamePadding) may own a
    # padding child module; treat the conv itself as the leaf and skip its children.
    conv_prefixes: list[str] = []
    for name, m in model.named_modules():
        if not name:
            continue
        if any(name.startswith(pfx + ".") for pfx in conv_prefixes):
            continue
        is_conv = isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        if is_conv:
            conv_prefixes.append(name)
        if is_conv or len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(leaf_hook(name)))

    def encoder_hook(m: nn.Module, inp: tuple, out: Any) -> None:
        for level, f in enumerate(out):
            encoder_levels.append(
                {
                    "level": level,
                    "shape": list(f.shape),
                    "stride": int(round(x.shape[-1] / f.shape[-1])),
                    "channels": int(f.shape[1]),
                }
            )

    handles.append(model.encoder.register_forward_hook(encoder_hook))

    for i, blk in enumerate(model.decoder.blocks):

        def dec_hook(m: nn.Module, inp: tuple, kwargs: dict, out: Any, i: int = i) -> None:
            skip = None
            if kwargs:
                skip = kwargs.get("skip_connection")
                if skip is None:
                    skip = kwargs.get("skip")
            if skip is None and len(inp) > 1:
                skip = inp[1]
            decoder_blocks.append(
                {
                    "block": i,
                    "in_shape": _shape(inp),
                    "skip_shape": _shape(skip) if skip is not None else None,
                    "out_shape": _shape(out),
                }
            )

        handles.append(blk.register_forward_hook(dec_hook, with_kwargs=True))

    with torch.no_grad():
        logits = model(x)

    for h in handles:
        h.remove()

    return {
        "layers": layers,
        "encoder_levels": encoder_levels,
        "decoder_blocks": decoder_blocks,
        "output_shape": list(logits.shape),
    }


def summarize(trace: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    """Aggregate counts that the explainer and README quote."""
    layers = trace["layers"]
    by_type = Counter(l["type"] for l in layers)
    convs = [l for l in layers if l["type"].startswith("Conv")]
    ksizes = Counter(tuple(l["kernel_size"]) for l in convs)
    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    head = sum(p.numel() for p in model.segmentation_head.parameters())
    return {
        "total_params": total,
        "encoder_params": enc,
        "decoder_params": dec,
        "head_params": head,
        "leaf_modules": len(layers),
        "modules_by_type": dict(by_type),
        "conv_layers": len(convs),
        "depthwise_convs": sum(1 for l in convs if l.get("depthwise")),
        "pointwise_convs": sum(1 for l in convs if l.get("pointwise")),
        "kernel_size_histogram": {"x".join(map(str, k)): v for k, v in ksizes.items()},
        "pooling_layers": {k: v for k, v in by_type.items() if "Pool" in k},
        "strided_convs": [
            {
                "name": l["name"],
                "kernel_size": l["kernel_size"],
                "stride": l["stride"],
                "depthwise": l.get("depthwise", False),
                "in_shape": l["in_shape"],
                "out_shape": l["out_shape"],
            }
            for l in convs
            if max(l["stride"]) > 1
        ],
        "upsample_layers": [
            {"name": l["name"], "scale_factor": l.get("scale_factor"), "mode": l.get("mode")}
            for l in layers
            if l["type"] == "Upsample"
        ],
        # Operations that are not modules (so hooks cannot see them), from the smp source:
        "functional_ops": {
            "decoder_upsampling": "F.interpolate(x, scale_factor=2, mode='nearest') at the start of every decoder block",
            "squeeze_excite_pooling": "F.adaptive_avg_pool2d(x, 1) (global average pooling) inside every MBConv block's squeeze-and-excitation",
            "head_activation": "torch.sigmoid applied to the logits at evaluation time (not part of the model)",
        },
    }


def stem_kernels(model: nn.Module, max_filters: int = 32) -> dict[str, Any]:
    """First-layer kernels: the only filters that look directly at pixels."""
    conv = model.encoder._conv_stem
    w = conv.weight.detach().cpu().numpy()[:max_filters]  # (out, in, kh, kw)
    return {
        "layer": "encoder._conv_stem",
        "shape": list(w.shape),
        "stride": _tolist(conv.stride),
        "kernels": np.round(w, 4).tolist(),
    }


# --------------------------------------------------------------------------- #
# Optional figures
# --------------------------------------------------------------------------- #
def load_image(path: str, size: int) -> torch.Tensor:
    """Load a grayscale image the way the notebook's dataset does (resize, divide by max)."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img.astype(np.float32), (size, size))
    img = img / (img.max() + 1e-8)
    return torch.tensor(img, dtype=torch.float32)[None, None]


def save_figures(model: nn.Module, x: torch.Tensor, figdir: Path) -> list[str]:
    """Save stem kernels, per-level encoder feature maps, decoder outputs and the prediction."""
    if plt is None:
        raise RuntimeError("matplotlib is required for --figdir")
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1) stem kernels
    w = model.encoder._conv_stem.weight.detach().cpu().numpy()
    n = w.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            k = w[i, 0]
            v = np.abs(k).max() + 1e-8
            ax.imshow(k, cmap="RdBu_r", vmin=-v, vmax=v)
    fig.suptitle(f"Stem kernels: {n} filters of {w.shape[2]}x{w.shape[3]} (red +, blue -)")
    p = figdir / "stem_kernels.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    # 2) encoder feature maps per level + decoder block outputs + prediction
    feats: dict[str, torch.Tensor] = {}
    hs = [model.encoder.register_forward_hook(lambda m, i, o: feats.update({f"enc{l}": f for l, f in enumerate(o)}))]
    for i, blk in enumerate(model.decoder.blocks):
        hs.append(blk.register_forward_hook(lambda m, i_, o, i=i: feats.update({f"dec{i}": o})))
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    for h in hs:
        h.remove()

    for key, f in feats.items():
        f = f[0].cpu().numpy()
        k = min(16, f.shape[0])
        cols = 8
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        for i, ax in enumerate(np.atleast_1d(axes).flat):
            ax.axis("off")
            if i < k:
                ax.imshow(f[i], cmap="viridis")
        fig.suptitle(f"{key}: {f.shape[0]} channels at {f.shape[1]}x{f.shape[2]} (first {k} shown)")
        p = figdir / f"featuremaps_{key}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x[0, 0].cpu(), cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(prob[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("sigmoid(logits)")
    for ax in axes:
        ax.axis("off")
    p = figdir / "prediction.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--size", type=int, default=256, help="input H=W used for the trace")
    ap.add_argument("--weights", default=None, help="trained .pth state_dict (optional)")
    ap.add_argument("--image", default=None, help="grayscale image to trace with (optional)")
    ap.add_argument("--out", default="docs/explainer/architecture.json")
    ap.add_argument("--summary", default="docs/explainer/torchinfo_summary.txt")
    ap.add_argument("--figdir", default=None, help="write kernel/feature-map PNGs here (needs --image to be meaningful)")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    model = build_model(args.encoder, args.in_channels, weights_path=args.weights)
    x = load_image(args.image, args.size) if args.image else torch.rand(1, args.in_channels, args.size, args.size)

    trace = trace_model(model, x)
    out = {
        "model": {
            "architecture": "Unet",
            "encoder": args.encoder,
            "encoder_weights": "checkpoint" if args.weights else "imagenet",
            "in_channels": args.in_channels,
            "classes": 1,
            "input_shape": list(x.shape),
            "library": f"segmentation_models_pytorch {smp.__version__}",
            "torch": torch.__version__,
        },
        "summary": summarize(trace, model),
        "encoder_levels": trace["encoder_levels"],
        "decoder_blocks": trace["decoder_blocks"],
        "output_shape": trace["output_shape"],
        "stem_kernels": stem_kernels(model),
        "layers": trace["layers"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    js_path = out_path.with_suffix(".js")
    js_path.write_text("window.ARCH = " + json.dumps(out) + ";\n")
    print(f"wrote {out_path} and {js_path} ({len(trace['layers'])} leaf modules)")

    try:
        from torchinfo import summary as ti_summary

        s = ti_summary(model, input_size=tuple(x.shape), depth=4, verbose=0, col_names=("input_size", "output_size", "num_params", "kernel_size"))
        Path(args.summary).write_text(str(s))
        print(f"wrote {args.summary}")
    except Exception as e:  # pragma: no cover
        print(f"torchinfo summary skipped: {e}")

    if args.figdir:
        for p in save_figures(model, x, Path(args.figdir)):
            print("wrote", p)

    s = out["summary"]
    print(
        f"\n{args.encoder} U-Net: {s['total_params']:,} params "
        f"(encoder {s['encoder_params']:,}, decoder {s['decoder_params']:,}, head {s['head_params']:,}); "
        f"{s['conv_layers']} convs ({s['depthwise_convs']} depthwise, {s['pointwise_convs']} 1x1); "
        f"kernel sizes {s['kernel_size_histogram']}; pooling layers {s['pooling_layers'] or 'none'}; "
        f"{len(s['strided_convs'])} strided convs; {len(s['upsample_layers'])} upsample layers"
    )


if __name__ == "__main__":
    main()
