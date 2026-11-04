# CLAUDE.md — working agreement for this repo

## What this project is

Exploratory research on mass (tumor) segmentation in mammograms (TOMPEI-CMMD).
Two success metrics, always reported together:

1. **Dice score** on a held-out test set (higher is better).
2. **Compute**: dataset bytes on disk, seconds per training epoch, peak GPU memory (lower is better).

The owner (Ella) is a student. Explain the *why* behind concepts and choices, not just the what.
Flag important design choices and anything that could push the research further **before** acting on it.
Routine analysis, refactors, and documentation: just do them.

## Repo layout

```
notebooks/   Colab notebooks (the original baseline lives here; keep outputs so results are visible)
src/         Reusable Python: data packaging, datasets, models, training, evaluation, visualization
docs/        README figures (docs/figures/), experiment log (docs/EXPERIMENTS.md), the CNN explainer (docs/explainer/)
experiments/ One folder per experiment: config.yaml, results.json, short README (no data, no weights)
```

Data and model weights are never committed (see .gitignore). Training runs on Colab GPU; local `.venv` is for CPU development and tests.

## Git workflow

- `main` is always in a presentable state. Work on branches: `feat/…`, `exp/…` (experiments), `chore/…`, `docs/…`, `fix/…`.
- Small, focused commits with imperative subjects (`Add breast-region crop`, `Cache dataset as uint8 npy`). Body explains why when it is not obvious.
- Merge to `main` with `git merge --no-ff` so each branch is one visible unit of work. (GitHub CLI is not installed; use PRs if `gh` becomes available.)
- Tag milestones that produce a reported number: `exp-YYYYMMDD-<slug>`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`

## Scientific standards

- Fixed random seeds. The train/val/test split is written to a file and reused by every experiment.
- Model and hyperparameter selection uses the **validation** set only. The test set is scored once per experiment.
- Report mean and spread (std or IQR) over test images, plus a paired test when comparing two models on the same images.
- Every experiment records compute alongside Dice: dataset size on disk, epoch time, peak GPU memory, parameter count.
- Append every completed experiment to `docs/EXPERIMENTS.md` (one row) with a link to its folder.
- Cite sources for any method taken from a paper or repo.

## Code conventions

- Library code goes in `src/` with docstrings and type hints; notebooks import from `src/`, they do not redefine it.
- Prefer plain PyTorch + `segmentation_models_pytorch`; add a dependency only if it earns its place.
- Keep scripts runnable from the repo root: `python -m src.<module> --help`.

## Roadmap (agreed 2026-11-04)

1. Package the dataset once as uint8 arrays at training resolution.
2. Crop to the breast region; test higher input resolution / lesion patches.
3. Validation split, Dice+BCE or Focal Tversky loss, augmentation.
4. Hyperparameter sweeps (preprocessing params, encoder size, LR, resolution).
5. Open-source models: mammography-pretrained encoders (Mammo-CLIP, VersaMammo); nnU-Net 2D as reference.
