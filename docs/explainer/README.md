# CNN explainer

`index.html` is an interactive walkthrough of the baseline network (U-Net, EfficientNet-B0 encoder): an architecture map you can click through, a convolution sandbox with the model's real first-layer kernels, a pooling comparison, the parameter breakdown, one MBConv block with its actual numbers, and the full 219-module layer table. It is inspired by [CNN Explainer](https://poloclub.github.io/cnn-explainer/).

Open it locally by double-clicking `index.html`, or via GitHub Pages once Pages is enabled for the `docs/` folder (Settings → Pages → Deploy from branch `main`, folder `/docs`), at `https://ellabellae.github.io/CNNdetection/explainer/`.

## Regenerating the data

`architecture.json` / `architecture.js` and `torchinfo_summary.txt` are produced from the live PyTorch model:

```bash
source .venv/bin/activate
python -m src.model_inspect --out docs/explainer/architecture.json
```

Add `--weights path/to/model_clahe.pth --image some_mammogram.png --figdir docs/figures/model` to trace a trained checkpoint on a real image and also save kernel and feature-map figures.
