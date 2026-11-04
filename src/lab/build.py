"""Build the Architecture Lab data bundle.

Runs every requested model over every requested sample image under every
requested preprocessing and resolution, and writes ``manifest.json`` plus
PNG sprites, masks and heat maps under ``--out``. CPU only; minutes, not hours.

Examples
--------
Default full run (needs raw sample images and the trained checkpoints)::

    python -m src.lab.build --out docs/lab/data

Smoke run on whatever is available::

    python -m src.lab.build --models unet_b0_imagenet --resolutions 256 --skip-missing-weights
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from . import bundle, masks, metrics, preprocess, trace as T
from .models import LINEUP, ModelSpec, build as build_model
from .receptive_field import RF, chain, upsample
from .saliency import grad_cam

THUMB = 256
ANNOT_DIR = "data/annotations/TOMPEI-CMMD_v01_20250123_new"


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
def load_sample(name: str, sample_dir: Path, annot_dir: Path) -> dict:
    """Load whatever exists for a sample: raw image, precomputed CLAHE image, annotation."""
    out: dict = {"name": name, "raw": None, "clahe_file": None, "ann": None}
    for cand in (sample_dir / "raw" / f"{name}.png", sample_dir / "raw" / f"{name}_raw.png"):
        if cand.exists():
            out["raw"] = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            break
    c = sample_dir / "clahe" / f"{name}_clahe.png"
    if c.exists():
        out["clahe_file"] = cv2.imread(str(c), cv2.IMREAD_GRAYSCALE)
    a = annot_dir / f"{name}_AnnotationFile.json"
    if a.exists():
        out["ann"] = masks.load_annotation(a)
    return out


def preprocessed_variants(sample: dict, names: list[str], log) -> dict[str, preprocess.Preprocessed]:
    """Every requested preprocessing that can be produced from what the sample has."""
    base = sample["raw"] if sample["raw"] is not None else sample["clahe_file"]
    if base is None:
        return {}
    h, w = base.shape
    gt = masks.rasterize(sample["ann"], h, w) if sample["ann"] is not None else np.zeros((h, w), np.uint8)
    out = {}
    for n in names:
        if sample["raw"] is None:
            if n == "clahe_median":
                out[n] = preprocess.Preprocessed(n, sample["clahe_file"], gt, {"source": "precomputed file (raw image unavailable)"})
            else:
                log(f"  [{sample['name']}] skipping {n}: raw image not available")
            continue
        out[n] = preprocess.apply(n, sample["raw"], gt)
    return out


# --------------------------------------------------------------------------- #
# Architecture (once per model)
# --------------------------------------------------------------------------- #
def receptive_fields(t: T.Trace) -> dict[str, dict]:
    """Theoretical RF per encoder level from the conv chain along the encoder; decoder inherits."""
    rf = RF(1.0, 1.0, 0.0)
    out: dict[str, dict] = {"enc0": asdict(rf)}
    seen_levels = {lv["shape"][2]: f"enc{lv['level']}" for lv in t.encoder_levels}
    for l in t.layers:
        if not l.group.startswith("encoder") or l.kernel_size is None:
            continue
        if l.in_shape and l.in_shape[2] == 1:   # squeeze-excite 1x1 convs on pooled 1x1 maps
            continue
        k, s, p = l.kernel_size[0], l.stride[0], l.padding[0] if l.padding else 0
        rf = chain([(k, s, p)], rf)
        if l.out_shape and l.out_shape[2] in seen_levels:
            out[seen_levels[l.out_shape[2]]] = asdict(rf)
    # decoder: receptive field of the deepest stage, grid refined by upsampling
    deepest = rf
    for b in t.decoder_blocks:
        factor = b["out_shape"][2] / t.encoder_levels[-1]["shape"][2]
        out[f"dec{b['block']}"] = asdict(upsample(deepest, factor))
    out["head"] = out.get(f"dec{len(t.decoder_blocks) - 1}", asdict(deepest))
    return out


def architecture_record(model, spec: ModelSpec, res: int, analytic_res: list[int], threads: int | None, log) -> dict:
    x = torch.rand(1, spec.in_channels, res, res)
    t = T.trace(model, x, capture=False)
    stem = model.encoder._conv_stem.weight.detach().cpu().numpy()
    rec = {
        "layers": T.layers_as_dicts(t),
        "stages": T.stage_list(t),
        "encoder_groups": T.encoder_groups_by_level(t),
        "decoder_blocks": t.decoder_blocks,
        "receptive_field": receptive_fields(t),
        "totals": t.totals(),
        "stem_kernels": {"shape": list(stem.shape), "kernels": np.round(stem[:32], 4).tolist()},
        "analytic": {},
        "timing": {},
    }
    for r in analytic_res:
        tr = T.trace(model, torch.rand(1, spec.in_channels, r, r), capture=False)
        tot = tr.totals()
        rec["analytic"][str(r)] = {
            "macs": tot["macs"], "elementwise": tot["elementwise"],
            "activation_bytes_total": tot["activation_bytes_total"], "activation_bytes_largest": tot["activation_bytes_largest"],
            "by_group": tot["by_group"], "input_bytes": preprocess.input_bytes(r, spec.in_channels),
            "stages": T.stage_list(tr),
        }
    return rec


# --------------------------------------------------------------------------- #
# One run = model x preprocessing x image x resolution
# --------------------------------------------------------------------------- #
def run_one(model, spec: ModelSpec, prep: preprocess.Preprocessed, name: str, res: int, out_dir: Path, rel: Path, k: int, tile: int) -> dict:
    x = torch.from_numpy(preprocess.to_input(prep.image, res, spec.input_convention))[None]
    gt = preprocess.resize_mask(prep.mask, res)
    t = T.trace(model, x, capture=True)
    prob = 1 / (1 + np.exp(-t.logits))
    pred = (prob > 0.5).astype(np.uint8)
    files: dict[str, str] = {}
    stages = []
    for st in T.stage_list(t):
        key = st["key"]
        if key == "head":
            continue
        feat = t.stage_tensors[key]
        sprite, meta = bundle.feature_sprite(feat, k=k, tile=tile)
        f = rel / f"{key}.png"
        bundle.write_png(out_dir / f, sprite)
        stages.append({**st, "sprite": str(f), **meta})
    files["prob"] = str(rel / "prob.png"); bundle.write_png(out_dir / files["prob"], bundle.heat_u8(prob, THUMB))
    files["pred"] = str(rel / "pred.png"); bundle.write_png(out_dir / files["pred"], bundle.mask_u8(pred, THUMB))
    # squeeze-excite gates: one strip image per block + a compact summary
    se = []
    if t.se_gates:
        strip = np.zeros((len(t.se_gates) * 8, max(g.shape[0] for g in t.se_gates.values())), np.uint8)
        for i, (g, vec) in enumerate(t.se_gates.items()):
            strip[i * 8:(i + 1) * 8, : vec.shape[0]] = bundle.gate_strip(vec)[0]
            top = np.argsort(-vec)[:8]
            se.append({"block": g, "channels": int(vec.shape[0]), "mean": float(vec.mean()), "min": float(vec.min()), "max": float(vec.max()),
                       "top_channels": [int(c) for c in top], "top_values": [round(float(vec[c]), 3) for c in top]})
        files["se_strip"] = str(rel / "se_gates.png"); bundle.write_png(out_dir / files["se_strip"], strip)
    # saliency
    sal: dict[str, str] = {}
    if gt.sum() > 0:
        for stage, label in (("encoder_last", "gt_encoder"), ("dec2", "gt_dec2")):
            cam = grad_cam(model, x, gt, stage)
            sal[label] = str(rel / f"cam_{label}.png"); bundle.write_png(out_dir / sal[label], bundle.heat_u8(cam, THUMB))
    missed = gt.sum() > 0 and not metrics.overlap(pred, gt)
    if (missed or gt.sum() == 0) and pred.sum() > 0:
        cam = grad_cam(model, x, pred, "encoder_last")
        sal["pred_encoder"] = str(rel / "cam_pred_encoder.png"); bundle.write_png(out_dir / sal["pred_encoder"], bundle.heat_u8(cam, THUMB))
    return {
        "model": spec.key, "image": name, "preprocessing": prep.name, "resolution": res,
        "native": prep.name == spec.native_preprocessing, "decoder_trained": spec.decoder_trained,
        "dice": metrics.dice(pred, gt), "iou": metrics.iou(pred, gt),
        "pred_area_px": int(pred.sum()), "gt_area_px": int(gt.sum()), "missed": bool(missed),
        "prob_max": float(prob.max()), "files": files, "stages": stages, "se": se, "saliency": sal,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace, log=print) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir, annot_dir = Path(args.sample_dir), Path(args.annotations_dir)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    # images
    if args.images:
        names = list(args.images)
        groups = {n: "custom" for n in names}
        baseline_scores: dict[str, dict] = {}
    else:
        sel = json.loads((sample_dir / "selected.json").read_text())
        names = [s["name"] for s in sel]
        groups = {s["name"]: s["group"] for s in sel}
        baseline_scores = {s["name"]: {"raw": s["raw"], "clahe": s["clahe"], "pca": s["pca"], "mass_area_px": s["mass_area"]} for s in sel}

    manifest: dict = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": f"{platform.machine()} CPU ({platform.system()})",
            "torch": torch.__version__, "threads": torch.get_num_threads(), "budget_mb": args.budget_mb,
            "resolutions": list(args.resolutions), "analytic_resolutions": list(args.analytic_resolutions),
            "preprocessings": list(args.preprocessings), "thumb": THUMB,
            "note": "Every number here is illustrative, computed on 8 sample images on a laptop CPU. docs/EXPERIMENTS.md is the source of reported results.",
            "licenses": {
                "TOMPEI-CMMD": "CC BY 4.0 - Kashiwada et al. 2025, https://doi.org/10.7937/WEZW-BH22",
                "Mammo-CLIP": "CC BY-NC-SA 4.0 - Ghosh et al. 2024, arXiv 2405.12255",
            },
        },
        "models": {}, "images": {}, "runs": [],
    }

    # samples + preprocessing (done once, shared by all models)
    variants: dict[str, dict[str, preprocess.Preprocessed]] = {}
    for n in names:
        s = load_sample(n, sample_dir, annot_dir)
        v = preprocessed_variants(s, list(args.preprocessings), log)
        if not v:
            log(f"  [{n}] no image available, skipping")
            continue
        variants[n] = v
        rel = Path("images") / n
        entry = {"group": groups.get(n, ""), "baseline": baseline_scores.get(n), "thumbs": {}, "gt": {}, "meta": {}}
        if s["ann"] is not None:
            entry["mass"] = masks.mass_summary(s["ann"])
        for pn, p in v.items():
            f = rel / f"{pn}.png"; bundle.write_png(out_dir / f, bundle.thumb(p.image, THUMB)); entry["thumbs"][pn] = str(f)
            g = rel / f"{pn}_gt.png"; bundle.write_png(out_dir / g, bundle.mask_u8(p.mask, THUMB)); entry["gt"][pn] = str(g)
            entry["meta"][pn] = {**p.meta, "source_shape": list(p.image.shape)}
        manifest["images"][n] = entry

    # models
    for key in args.models:
        spec = LINEUP[key] if key in LINEUP else _adhoc_spec(key)
        try:
            model = build_model(spec, require_weights=not args.skip_missing_weights)
        except FileNotFoundError as e:
            log(f"  skipping {key}: {e}")
            continue
        log(f"model {key}: {spec.title}")
        arch = architecture_record(model, spec, args.resolutions[0], list(args.analytic_resolutions), args.threads, log)
        for res in args.resolutions:
            x = torch.rand(1, spec.in_channels, res, res)
            arch["timing"][str(res)] = {**T.time_forward(model, x, warmup=2, runs=5), "stages": T.time_stages(model, x, runs=3), "machine": manifest["meta"]["machine"]}
            log(f"  timing {res}: {arch['timing'][str(res)]['median_s'] * 1000:.0f} ms")
        manifest["models"][key] = {**asdict(spec), **arch}
        model_res = list(args.resolutions) if spec.decoder_trained else [r for r in args.resolutions if r in args.untrained_resolutions]
        for n, v in variants.items():
            for pn, p in v.items():
                if not args.cross_apply and pn != spec.native_preprocessing and pn not in args.always_preprocessings:
                    continue
                for res in model_res:
                    rel = Path("runs") / key / n / pn / str(res)
                    r = run_one(model, spec, p, n, res, out_dir, rel, args.max_channels, args.tile)
                    manifest["runs"].append(r)
                    log(f"  {n:16s} {pn:13s} {res:4d}  dice={r['dice']:.3f}  pred={r['pred_area_px']:6d}px gt={r['gt_area_px']:6d}px" + ("  MISSED" if r["missed"] else ""))
        del model

    manifest = _round_floats(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    (out_dir / "manifest.js").write_text("window.LAB = " + json.dumps(manifest, separators=(",", ":")) + ";\n")
    n = bundle.check_budget(out_dir, args.budget_mb)
    log(f"bundle: {n / 1e6:.1f} MB in {out_dir} (budget {args.budget_mb} MB), {len(manifest['runs'])} runs")
    return manifest


def _round_floats(o, nd: int = 4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round_floats(v, nd) for v in o]
    return o


def _adhoc_spec(key: str) -> ModelSpec:
    """``random:<encoder>:<in_channels>`` builds an untrained model (used by tests and smoke runs)."""
    if key.startswith("random:"):
        _, enc, ch = key.split(":")
        conv = "mammoclip" if int(ch) == 3 else "baseline"
        return ModelSpec(key, f"random {enc}", enc, int(ch), conv, "random", None, "raw", False, "random init", "n/a")
    raise KeyError(f"unknown model {key!r}; known: {sorted(LINEUP)}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(LINEUP), help=f"lineup keys {sorted(LINEUP)} or random:<encoder>:<channels>")
    ap.add_argument("--images", nargs="*", default=None, help="sample names (default: data/samples/selected.json)")
    ap.add_argument("--preprocessings", nargs="+", default=list(preprocess.PREPROCESSING_NAMES))
    ap.add_argument("--always-preprocessings", nargs="*", default=["raw", "clahe_median"], help="preprocessings every trained model is run on even without --cross-apply")
    ap.add_argument("--cross-apply", action="store_true", help="run every trained model on every preprocessing")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--untrained-resolutions", nargs="+", type=int, default=[256], help="resolutions run for models whose decoder is untrained (their masks are noise; one resolution is enough)")
    ap.add_argument("--analytic-resolutions", nargs="+", type=int, default=[256, 512, 1024])
    ap.add_argument("--out", default="docs/lab/data")
    ap.add_argument("--sample-dir", default="data/samples")
    ap.add_argument("--annotations-dir", default=ANNOT_DIR)
    ap.add_argument("--budget-mb", type=float, default=25.0)
    ap.add_argument("--max-channels", type=int, default=12)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--skip-missing-weights", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        build(args)
    except bundle.BudgetExceeded as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
