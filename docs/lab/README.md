# Architecture Lab

`index.html` is the five-tab lab specified in [`../specs/architecture-lab.md`](../specs/architecture-lab.md): Anatomy (architecture map, convolution sandbox with real kernels, layer table), Simulate (model × preprocessing × image × stage: feature maps, SE gates, receptive fields, Grad-CAM), Compare (two models or preprocessings side by side), Compute (parameters, MACs, activation memory, input bytes, CPU latency with a resolution slider) and Glossary.

It reads `data/manifest.js`, produced by the generator:

```bash
source .venv/bin/activate
python -m src.lab.build --out docs/lab/data            # full default run
python -m src.lab.build --help                          # every option
```

The generator needs, under `data/` (not committed):

- `data/samples/raw/<name>.png` — raw 8-bit mammograms for the eight selected images (`data/samples/selected.json`)
- `data/samples/clahe/<name>_clahe.png` — the team's CLAHE versions (optional; computed from raw if absent)
- `data/annotations/TOMPEI-CMMD_v01_20250123_new/*.json` — the annotation archive
- `data/weights/model_raw.pth`, `data/weights/model_clahe.pth` — the team's trained checkpoints
- `data/weights/mammoclip_b2_encoder.pth`, `data/weights/mammoclip_b5_encoder.pth` — image encoders extracted from the published Mammo-CLIP checkpoints (`shawn24/Mammo-CLIP` on Hugging Face; see `src/lab/models.py`)

Missing weights are skipped with `--skip-missing-weights`; a sample with only a CLAHE file is run on that preprocessing only. The bundle is refused if it exceeds `--budget-mb` (default 25).

Every number the page shows is illustrative (eight images, laptop CPU). `docs/EXPERIMENTS.md` remains the only source of reported results.
