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

1. **Track B — feature set. DONE (1 seed) — `body_raw_hands_derived` wins:**

   | features | dim | macro-F1 | real-eval | acc |
   |---|---|---|---|---|
   | raw | 152 | 0.872 | 0.846 | 0.914 |
   | derived | 39 | 0.724 | 0.673 | 0.833 |
   | both *(current lock)* | 189 | 0.872 | 0.847 | 0.915 |
   | **body_raw_hands_derived** | **105** | **0.896** | **0.876** | 0.919 |

   `body_raw_hands_derived` = raw body coords + presence flags + ALL derived
   (drops the raw hand coords, keeps only the scale-invariant derived hand
   features). It beats `both` by ~3 real-eval points AND is smaller — the raw
   hand coords carry the MediaPipe hand-domain-gap noise; the derived hand
   features are the clean signal.

   **Action:** re-run the 5-seed stability check with
   `--features body_raw_hands_derived` (mirror `docs/phase4_baseline.md`'s
   table). If the mean holds ≥ ~0.86 real-eval, switch the MLP lock to that
   feature mode: retrain `mlp.joblib`, update `train.py` (make
   `body_raw_hands_derived` the MLP default), regenerate the report, update
   `docs/phase4_baseline.md`. Then also re-check LightGBM at this feature mode
   for the record.

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
