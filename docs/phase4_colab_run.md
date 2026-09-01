# Phase 4 Track C — Colab training run (in progress)

**State as of this session:** the colab-mcp bridge is set up and connected. A
Colab scratch notebook is running `python -m classifier.train --model all` on
the 12-class + val-split dataset (12 GB RAM, 2 CPUs — slow, ~20 min).

## If resuming in a fresh session

1. `mcp__colab-mcp__open_colab_browser_connection` (needs a browser click to
   confirm) — the Colab **kernel keeps running server-side** even across a
   Claude restart, so the training may already be done.
2. `mcp__colab-mcp__get_cells(includeOutputs=true)` — read the training cell
   (index 4) and the eval cell.
3. Notebook cells already added:
   - 0: clone repo + `git pull` + resource check
   - 1: `pip install lightgbm catboost seaborn`
   - 2: `git pull` + `python -m pipeline.build_dataset` (train 52738 / val 2357 /
        test 1254 — matches local)
   - 4: `python -m classifier.train --model all`
   - **still to add**: `python -m classifier.evaluate --all` then a cell that
     `print`s `results/phase4/model_comparison.json` + each `*.eval.json` so the
     numbers come back as cell output (no file download needed).
4. To get the winning model's `.joblib` back: either retrain it locally
   (LGBM ~100 s) or base64 it out of a Colab cell.

## Local state (all committed, pushed)

- `data/features/*.npz` now committed (was git-ignored) — the s01/s02 test set.
- rf / lgbm / logreg trained locally on the val-split dataset:
  - **LGBM macro-F1 0.888 / real-eval 0.866 / acc 0.923 / ECE 0.036 / 31 MB** —
    clear leader
  - RF 0.823 / 0.788 / 0.146 / 389 MB (laying collapsed to 0.41)
  - LogReg 0.667 / 0.61 — the linear floor
- `results/phase4/model_comparison.json` + `results/phase4/fig/` hold the local
  3-model comparison.

## Next after the Colab numbers land

1. Regenerate a Phase 4 report (new script `scripts/phase4_report.py` — model
   comparison table + per-class F1 + calibration figures, like the Phase 3 one).
2. Pick + lock one model → write `docs/phase4_baseline.md` with the rationale.
   LGBM leads; the open questions are whether CatBoost calibrates better and
   whether the MLP's <1 MB size + temperature scaling makes it competitive.
3. Then Track D (augmentation strength on the val split) and Track E (per-class
   confidence thresholds on the val split).
