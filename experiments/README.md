# experiments/

One folder per experiment, named `YYYYMMDD_<slug>/`, containing:

- `config.yaml` — every setting needed to reproduce the run (seed, split file, resolution, model, loss, optimizer, epochs, hardware)
- `results.json` — test metrics (mean, std, per-image if small) and compute (epoch seconds, peak GPU MB, dataset bytes, param count)
- `README.md` — two or three sentences: what changed, what happened, what to try next

No data, weights, or large logs. Summarize each finished experiment as one row in `docs/EXPERIMENTS.md`.
