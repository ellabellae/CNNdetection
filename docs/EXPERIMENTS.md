# Experiment log

One row per completed experiment. Dice/IoU are test-set means (n = test images). Compute columns are measured on the hardware named in the experiment folder.

| ID | Date | Branch / folder | Change vs. baseline | Input | Loss | Dice | IoU | Epoch time | Peak GPU mem | Dataset on disk | Notes |
|----|------|-----------------|---------------------|-------|------|------|-----|------------|--------------|-----------------|-------|
| B0 | 2026-04 | `notebooks/CNN_tumor.ipynb` | baseline, raw | 256² | BCE | 0.521 | 0.416 | n/a | n/a | 1013 PNG, 2294×1914 | Colab GPU, 20 epochs, batch 4 |
| B1 | 2026-04 | `notebooks/CNN_tumor.ipynb` | baseline, CLAHE + median | 256² | BCE | 0.614 | 0.504 | n/a | n/a | same | best baseline |
| B2 | 2026-04 | `notebooks/CNN_tumor.ipynb` | baseline, PCA 0.9 | 256² | BCE | 0.584 | 0.475 | n/a | n/a | same | |
