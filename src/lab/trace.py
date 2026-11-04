"""Trace one forward pass: per-layer cost and shapes, stage outputs, SE gates, timing.

Cost model
----------
* multiply-accumulates (MACs) for a convolution = output elements × (C_in / groups) × k_h × k_w;
  for a linear layer = in × out; normalisation and activations are counted as one
  elementwise op per element and reported separately as ``elementwise``.
* ``out_bytes`` is the float32 size of each module's output. Summing them gives the
  bytes of every intermediate tensor the forward pass produces (an upper bound on
  activation memory; the largest single tensor is reported too).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LayerRecord:
    order: int
    name: str
    group: str
    type: str
    in_shape: list[int] | None
    out_shape: list[int] | None
    params: int
    macs: int
    elementwise: int
    out_bytes: int
    kernel_size: list[int] | None = None
    stride: list[int] | None = None
    padding: list[int] | None = None
    groups: int | None = None
    depthwise: bool = False
    pointwise: bool = False


def _tolist(v: Any) -> list[int] | None:
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        return [int(x) for x in v]
    return [int(v), int(v)]


def _shape(t: Any) -> list[int] | None:
    if isinstance(t, torch.Tensor):
        return list(t.shape)
    if isinstance(t, (tuple, list)) and t and isinstance(t[0], torch.Tensor):
        return list(t[0].shape)
    return None


def _numel(shape: list[int] | None) -> int:
    return int(np.prod(shape)) if shape else 0


def group_of(name: str) -> str:
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


def cost_of(m: nn.Module, in_shape: list[int] | None, out_shape: list[int] | None) -> tuple[int, int]:
    """(macs, elementwise) for one leaf module given its tensor shapes."""
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0] * m.kernel_size[1]
        return _numel(out_shape) * (m.in_channels // m.groups) * k, 0
    if isinstance(m, nn.ConvTranspose2d):
        k = m.kernel_size[0] * m.kernel_size[1]
        return _numel(in_shape) * (m.out_channels // m.groups) * k, 0
    if isinstance(m, nn.Linear):
        return m.in_features * m.out_features, 0
    if isinstance(m, (nn.BatchNorm2d, nn.ReLU, nn.SiLU, nn.Sigmoid, nn.GELU, nn.LeakyReLU)) or type(m).__name__ in ("MemoryEfficientSwish", "Swish"):
        return 0, _numel(out_shape)
    return 0, 0


def describe(m: nn.Module) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        d.update(kernel_size=_tolist(m.kernel_size), stride=_tolist(m.stride), padding=_tolist(m.padding), groups=m.groups)
        d["depthwise"] = m.groups > 1 and m.groups == m.in_channels
        d["pointwise"] = tuple(m.kernel_size) == (1, 1)
    elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
        d.update(kernel_size=_tolist(m.kernel_size), stride=_tolist(m.stride), padding=_tolist(m.padding))
    return d


@dataclass
class Trace:
    layers: list[LayerRecord]
    encoder_levels: list[dict]          # level, shape, stride, channels
    decoder_blocks: list[dict]          # block, in/skip/out shapes
    output_shape: list[int]
    stage_tensors: dict[str, np.ndarray] = field(default_factory=dict)   # "enc0".."enc5", "dec0"..,"head"
    se_gates: dict[str, np.ndarray] = field(default_factory=dict)        # block group -> (C,)
    logits: np.ndarray | None = None
    model_params: int = 0

    def totals(self) -> dict[str, Any]:
        macs = sum(l.macs for l in self.layers)
        ew = sum(l.elementwise for l in self.layers)
        by_group: dict[str, dict[str, int]] = {}
        for l in self.layers:
            g = by_group.setdefault(l.group, {"params": 0, "macs": 0, "out_bytes": 0})
            g["params"] += l.params
            g["macs"] += l.macs
            g["out_bytes"] += l.out_bytes
        traced = sum(l.params for l in self.layers)
        return {
            "params": self.model_params,
            "params_traced": traced,
            "params_unused": self.model_params - traced,   # e.g. EfficientNet's _conv_head, which smp never runs
            "macs": macs,
            "elementwise": ew,
            "activation_bytes_total": sum(l.out_bytes for l in self.layers),
            "activation_bytes_largest": max((l.out_bytes for l in self.layers), default=0),
            "by_group": by_group,
        }


def _leaf_modules(model: nn.Module):
    """Yield (name, module) treating any Conv2d subclass as a leaf (smp's padded convs own a child)."""
    conv_prefixes: list[str] = []
    for name, m in model.named_modules():
        if not name or any(name.startswith(p + ".") for p in conv_prefixes):
            continue
        is_conv = isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        if is_conv:
            conv_prefixes.append(name)
        if is_conv or len(list(m.children())) == 0:
            yield name, m


def trace(model: nn.Module, x: torch.Tensor, capture: bool = True) -> Trace:
    """Run ``model(x)`` under hooks and return the trace. ``capture=False`` skips tensor copies."""
    layers: list[LayerRecord] = []
    enc_levels: list[dict] = []
    dec_blocks: list[dict] = []
    stage_tensors: dict[str, np.ndarray] = {}
    se_gates: dict[str, np.ndarray] = {}
    handles = []
    counter = {"i": 0}

    def leaf_hook(name: str):
        def hook(m, inp, out):
            counter["i"] += 1
            ins, outs = _shape(inp), _shape(out)
            macs, ew = cost_of(m, ins, outs)
            rec = LayerRecord(
                order=counter["i"], name=name, group=group_of(name), type=type(m).__name__.replace("Conv2dStaticSamePadding", "Conv2d"),
                in_shape=ins, out_shape=outs, params=sum(p.numel() for p in m.parameters(recurse=False)),
                macs=macs, elementwise=ew, out_bytes=4 * _numel(outs), **describe(m),
            )
            layers.append(rec)
            if name.endswith("_se_expand") and capture:
                se_gates[group_of(name)] = torch.sigmoid(out.detach()).flatten().cpu().numpy()
        return hook

    for name, m in _leaf_modules(model):
        handles.append(m.register_forward_hook(leaf_hook(name)))

    def enc_hook(m, inp, out):
        for level, f in enumerate(out):
            enc_levels.append({"level": level, "shape": list(f.shape), "stride": int(round(x.shape[-1] / f.shape[-1])), "channels": int(f.shape[1])})
            if capture:
                stage_tensors[f"enc{level}"] = f.detach()[0].cpu().numpy()

    handles.append(model.encoder.register_forward_hook(enc_hook))

    for i, blk in enumerate(model.decoder.blocks):
        def dec_hook(m, inp, kwargs, out, i=i):
            skip = kwargs.get("skip_connection") if kwargs else None
            if skip is None:
                skip = kwargs.get("skip") if kwargs else None
            if skip is None and len(inp) > 1 and isinstance(inp[1], torch.Tensor):
                skip = inp[1]
            dec_blocks.append({"block": i, "in_shape": _shape(inp), "skip_shape": _shape(skip) if skip is not None else None, "out_shape": _shape(out)})
            if capture:
                stage_tensors[f"dec{i}"] = out.detach()[0].cpu().numpy()
        handles.append(blk.register_forward_hook(dec_hook, with_kwargs=True))

    with torch.no_grad():
        logits = model(x)
    for h in handles:
        h.remove()
    t = Trace(layers=layers, encoder_levels=enc_levels, decoder_blocks=dec_blocks, output_shape=list(logits.shape), stage_tensors=stage_tensors, se_gates=se_gates,
              model_params=sum(p.numel() for p in model.parameters()))
    if capture:
        t.stage_tensors["head"] = logits.detach()[0].cpu().numpy()
        t.logits = logits.detach()[0, 0].cpu().numpy()
    return t


def stage_list(t: Trace) -> list[dict]:
    """Ordered stages for the page: encoder levels, decoder blocks, head."""
    stages = []
    for lv in t.encoder_levels:
        stages.append({"key": f"enc{lv['level']}", "kind": "input" if lv["level"] == 0 else "encoder", "shape": lv["shape"][1:], "stride": lv["stride"]})
    for b in t.decoder_blocks:
        stages.append({"key": f"dec{b['block']}", "kind": "decoder", "shape": b["out_shape"][1:], "skip_shape": (b["skip_shape"][1:] if b["skip_shape"] else None)})
    stages.append({"key": "head", "kind": "head", "shape": t.output_shape[1:]})
    return stages


def encoder_groups_by_level(t: Trace) -> dict[str, list[str]]:
    """Map each encoder level key to the module groups whose outputs have that spatial size."""
    out: dict[str, list[str]] = {}
    for lv in t.encoder_levels:
        if lv["level"] == 0:
            continue
        size = lv["shape"][2]
        groups: list[str] = []
        for l in t.layers:
            if l.group.startswith("encoder") and l.out_shape and l.out_shape[2] == size and l.group not in groups:
                groups.append(l.group)
        out[f"enc{lv['level']}"] = groups
    return out


def time_forward(model: nn.Module, x: torch.Tensor, warmup: int = 2, runs: int = 5, threads: int | None = None) -> dict[str, float]:
    """Median wall-clock seconds of a forward pass (CPU, synchronous)."""
    if threads:
        torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "runs": runs, "threads": torch.get_num_threads()}


def time_stages(model: nn.Module, x: torch.Tensor, runs: int = 3) -> dict[str, float]:
    """Median seconds spent inside the encoder, each decoder block and the head."""
    marks: dict[str, list[float]] = {}
    starts: dict[str, float] = {}
    handles = []

    def pre(key):
        def h(m, inp, kwargs=None):
            starts[key] = time.perf_counter()
        return h

    def post(key):
        def h(m, inp, out):
            marks.setdefault(key, []).append(time.perf_counter() - starts[key])
        return h

    targets = {"encoder": model.encoder, "head": model.segmentation_head}
    for i, b in enumerate(model.decoder.blocks):
        targets[f"dec{i}"] = b
    for key, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(pre(key)))
        handles.append(mod.register_forward_hook(post(key)))
    with torch.no_grad():
        model(x)  # warm
        for k in list(marks):
            marks[k] = []
        for _ in range(runs):
            model(x)
    for h in handles:
        h.remove()
    return {k: statistics.median(v) for k, v in marks.items()}


def layers_as_dicts(t: Trace) -> list[dict]:
    return [asdict(l) for l in t.layers]
