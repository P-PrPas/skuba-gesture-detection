# Phase 4 — resume notes

## Done (committed)

- **Classifier LOCKED = MLP** (`classifier/mlp.py`, seed 0). `train.py` default is
  `mlp`. 5-seed real-eval 0.857 ± 0.016 (tied with LightGBM). Chosen for
  thresholdable confidence (0.85 correct / 0.49 wrong) + 0.3 MB.
  `docs/phase4_baseline.md`.
- Track C comparison: `results/phase4/classifier_report.docx`,
  `results/phase4/model_comparison.json`, `results/phase4/fig/`.
- `data/dataset/val.npz` — 15% held-out slice of each external class's clean
  pre-augment rows (`VAL_FRAC` in `build_dataset.py`). For early stopping /
  temperature scaling / threshold tuning. Never touch `test.npz`.
- `classifier/evaluate.py` — model-agnostic harness. `--all` scores every
  `models/*.joblib`; `--split {test,val}`.
- s01/s02 features (`data/features/*.npz`) now committed (were git-ignored).
- colab-mcp bridge is set up in this project's local MCP config. It works but
  the free Colab tier is 2 weak CPUs — RandomForest took 7 min there, CatBoost
  and HGB each stalled past 20 min. Not useful for training; fine for light
  scripting.

## Track C leftovers (optional — decision already made)

- CatBoost / HistGradientBoosting / RBF-SVM never completed (Colab too slow).
  Same GBDT family as LightGBM — they would not beat the locked MLP on the
  deciding metric (confidence separation). Run on a real machine only if the
  lock is re-opened.

## Next — Track B / D / E (a fresh session does these cleanly)

1. **Track B — feature set.** MLP `--features` sweep was running when this
   session ended; partial result in a scratch log — `raw` (152-d) tied `both`
   (189-d) at real-eval 0.846. Re-run:
   ```
   for f in raw derived both body_raw_hands_derived; do
     python -m classifier.train --model mlp --features $f
     python -m classifier.evaluate --model mlp
   done
   ```
   If `raw` holds, switch the MLP to `raw` (smaller input, the net learns the
   derived interactions itself) and retrain + re-lock.

2. **Track D — augmentation strength.** `AugParams` in `features/augment.py`.
   MLP's weak classes are `mini_heart` (0.73) and `raise_*_hand` (0.76). Tune
   `n_per_sample`, `rot_deg`, `coord_noise_std`, the `mini_heart` elevation
   range, etc. against `val.npz` (not test). Re-run the harness after each
   change — a fix for one class must not break another.

3. **Track E — per-class confidence thresholds.** On `val.npz`, per class, find
   the max-probability threshold below which the frame goes to `idle/unknown`.
   Wire into the Phase 5 smoothing/fallback. `idle` currently over-triggers
   (precision ~0.47).

4. Regenerate `results/phase4/classifier_report.docx` from the harness outputs;
   update `docs/phase4_baseline.md` if the pick config changes.
