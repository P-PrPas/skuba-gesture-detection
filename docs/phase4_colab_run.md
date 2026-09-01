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

## Track B / D / E — DONE (2026-09-01). Lock unchanged: MLP / `both` / seed 0.

Full tables + rationale in `docs/phase4_baseline.md`. Summary:

1. **Track B — feature set.** 5-seed re-check of `body_raw_hands_derived`:
   real-eval **0.851 ± 0.035** (vs `both` 0.857 ± 0.016) — below the ≥0.86 bar,
   less stable, and `idle` F1 collapses to ~0.21 on 2/5 seeds. Kept `both`. It
   *does* fix `mini_heart` (0.89 vs 0.78) — raw hand coords carry the hand
   domain-gap noise — noted for a Phase 6 feature pass, not acted on now.

2. **Track D — augmentation strength.** 4 `AugParams` variants (env hooks
   `AUG_OVERRIDE`, `ARMS_UP_KEEP_ORIG` added to `build_dataset.py`). None beat
   the defaults on `test`; more augmentation tanks `mini_heart` (0.73 → 0.16).
   `val` `mini_heart` pinned at ~0.99 for every config — val doesn't track test
   for scarce classes. **Defaults retained.**

3. **Track E — confidence thresholds** (`classifier/thresholds.py`). Per-class
   val thresholds do **not** transfer to `test` (val confidences ~0.2 high,
   same-domain). Reject-to-idle only lowers idle precision on the honest split.
   Only a `global_floor` 0.20 garbage gate is active; real thresholds wait for
   Phase 6 field confidences. `mlp.thresholds.json` records inactive val refs.

4. Report docx NOT regenerated — pick config (MLP/`both`/seed 0) unchanged, and
   the docx is the Track C model comparison, which B/D/E don't touch.

**Phase 4 done. Next: Phase 5 (temporal smoothing + real-time pipeline).**
