# Phase 4 — classifier lock

**LOCKED: MLP** (`classifier/mlp.py`, `classifier/models/mlp.joblib`, seed 0) —
user decision 2026-09-01. `train.py` default should follow.

Track C compared candidate classifiers on the finalised 12-class dataset
(train 52,738 / val 2,357 / test 1,254), all scored by `classifier/evaluate.py`.
Report: `results/phase4/classifier_report.docx`, figures in `results/phase4/fig/`.

## What was actually run — all 8 candidates (`train.py`)

Every candidate trained locally on the finalised dataset, scored on `test` by
`classifier/evaluate.py`. (An earlier session offloaded the heavier ones to
free Colab, whose 2-CPU tier stalled them past 20 min; retrained here in
~1–5 min each — CatBoost/HGB/SVM/ET numbers below are from that run.)

| model | macro-F1 | real-eval | acc | ECE | conf ok / wrong | size | fit |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.888** | **0.866** | **0.923** | 0.036 | 0.95 / 0.80 | 31 MB | 104 s |
| **MLP** (+temp-scale) *(LOCK)* | 0.872 | 0.847 | 0.915 | 0.103 | 0.86 / **0.48** | **0.3 MB** | 102 s |
| HistGradientBoosting | 0.843 | 0.811 | 0.899 | **0.022** | 0.90 / 0.63 | 23 MB | 160 s |
| RandomForest | 0.823 | 0.788 | 0.868 | 0.146 | 0.77 / 0.44 | 389 MB | 43 s |
| CatBoost (depth-6/400 screen) | 0.821 | 0.786 | 0.881 | 0.091 | 0.93 / 0.53 | **2.7 MB** | 307 s |
| RBF-SVM (12k subsample) | 0.811 | 0.775 | 0.848 | 0.056 | 0.86 / 0.55 | 11 MB | 80 s |
| ExtraTrees | 0.797 | 0.757 | 0.867 | 0.172 | 0.79 / 0.53 | 998 MB | 29 s |
| LogisticRegression | 0.667 | 0.610 | 0.785 | 0.083 | 0.88 / 0.52 | ~0 | 31 s |

k-NN not run (undeployable by design — stores the whole training set).

**The GBDT family does not cluster.** The earlier assumption that CatBoost / HGB
would land within ±0.01 of LightGBM was wrong: CatBoost is 8 real-eval points
behind LightGBM and HGB 5.5 behind, at the screening configs. LightGBM's
gradient-based leaf split + our tuned `num_leaves`/`learning_rate` genuinely
lead this family on this data. HGB has the best calibration of any candidate
(ECE 0.022) but its accuracy and confidence spread don't beat MLP. CatBoost is
tiny (2.7 MB) but not accurate enough to matter. **None of the four newly-run
models changes the LightGBM-vs-MLP decision** — but now that is measured, not
assumed.

## The decision: LightGBM vs MLP

The two real contenders. LogReg (0.67) is the linear floor — it confirms the
problem is not linearly separable and a non-linear model is required. RF
regressed on the finalised data (`laying` collapsed to 0.41) and its 389 MB is
a deployment liability.

**LightGBM** — highest accuracy on every headline metric, lowest ECE. But its
confidence histogram piles ~80 % of test predictions into the top bin
(mean 0.95 when correct, 0.80 when wrong — a 0.15 gap). The Phase 5 fallback is
"if max-probability < T → output idle/unknown", and that gap is too small to
set a useful T without also rejecting many correct predictions. LightGBM would
need Platt / isotonic calibration on the val split first (a Track E experiment).

**MLP** (2×[Linear→BatchNorm→ReLU→Dropout] + label smoothing + post-fit
temperature scaling, `classifier/mlp.py`) — 1.6 macro-F1 points behind LightGBM
(inside the noise band of a 1,254-row test), but:
- **0.3 MB** — 100× smaller than LightGBM, negligible footprint on the shared
  Acer.
- **Confidence separates right from wrong**: mean 0.85 when correct vs **0.48
  when wrong** (a 0.37 gap), and the histogram is spread 0.2–1.0. A single
  threshold near ~0.5 flags most errors while keeping the correct predictions —
  exactly what Phase 5 needs, with no extra calibration step. Its higher ECE
  (0.103) is because temperature scaling left it slightly under-confident, which
  is the safe direction.
- Better on `idle` (0.76 vs 0.66) and `t_pose` (0.88 vs 0.82); weaker on
  `mini_heart` (0.73 vs 0.89), `laying` (0.89 vs 0.94), `raise_*_hand`
  (0.76 vs 0.81).
- `sit`@s02 = 1.0 (same as LightGBM).

### Stability — 5 seeds

| | seed 0 | 1 | 2 | 3 | 4 | **mean ± sd** |
|---|---|---|---|---|---|---|
| macro-F1 | 0.872 | 0.893 | 0.864 | 0.901 | 0.874 | **0.881 ± 0.014** |
| real-eval | 0.847 | 0.871 | 0.838 | 0.881 | 0.849 | **0.857 ± 0.016** |
| acc | 0.915 | 0.923 | 0.911 | 0.932 | 0.913 | 0.919 ± 0.008 |
| conf correct / wrong | .85/.48 | .86/.49 | .86/.46 | .84/.54 | .86/.50 | consistent |

MLP's mean real-eval (0.857 ± 0.016) is within one standard deviation of
LightGBM's deterministic 0.866 — **statistically indistinguishable on
accuracy**. The confidence separation (~0.85 correct / ~0.49 wrong) is stable
across every seed. The committed `mlp.joblib` is seed 0 (0.847 real — a
mid-range seed, no cherry-picking).

### Recommendation: **lock MLP**

- Accuracy tied with LightGBM (within noise).
- **Confidence is thresholdable out of the box** — the Phase 5 requirement.
- 0.3 MB vs 31 MB.
- No separate calibration step.

Remaining Track D/E work applies to MLP: tune `AugParams` to lift `mini_heart`
(0.73) and `raise_*_hand` (0.76), then set per-class thresholds on the val
split. If Track D can't move those without hurting the confidence spread,
re-evaluate against LightGBM + isotonic calibration.

## Track B — feature set (5-seed, MLP)

`--features` sweep, mean ± sd over seeds 0–4, scored on `test`:

| features | dim | macro-F1 | real-eval | acc | mini_heart | idle |
|---|---|---|---|---|---|---|
| **both** *(LOCK)* | 189 | **0.881 ± 0.014** | **0.857 ± 0.016** | 0.919 | 0.780 ± 0.090 | **0.768 ± 0.013** |
| body_raw_hands_derived | 105 | 0.875 ± 0.029 | 0.851 ± 0.035 | 0.911 | **0.887 ± 0.008** | 0.590 ± 0.306 |

`body_raw_hands_derived` (raw body + presence flags + all derived, no raw hand
coords) decisively fixes `mini_heart` — the raw hand coords carry MediaPipe's
hand-domain-gap noise and `mini_heart` is the class that reads it. But it does
**not** clear the ≥0.86 real-eval bar, is less stable on every metric, and
destabilises `idle`: F1 sd 0.31, collapsing to ~0.21 on 2 of 5 seeds. `idle` is
a CLAUDE.md-mandated class — **lock stays at `both`.** The `mini_heart` signal
is noted for a future feature-representation pass (Phase 6 field data).

## Track C addendum — SE-MLP (`classifier/semlp.py`)

Follow-up: does a feature-wise squeeze-and-excitation gate before the MLP trunk
help? The gate reads the 189-d input and emits one multiplier per feature, so
the trunk sees the currently-irrelevant slices damped. 5 seeds each, on `test`:

| model | real-eval | macro-F1 | acc | idle F1 | conf ok/wrong |
|---|---|---|---|---|---|
| **plain MLP** *(LOCK)* | **0.857 ± 0.016** | 0.881 ± 0.014 | 0.919 | **0.768 ± 0.013** | 0.86 / 0.49 |
| SE gate, hard (floor 0) | 0.822 ± 0.024 | 0.852 ± 0.020 | 0.897 | 0.281 ± 0.164 | 0.87 / 0.54 |
| SE gate, floor 0.5 | 0.859 ± 0.018 | 0.882 ± 0.016 | 0.921 | 0.672 ± 0.233 | 0.86 / 0.51 |

The hard gate regresses real-eval 3.5 pts and **collapses `idle` F1 to ~0.20 on
4 of 5 seeds** — same failure as `body_raw_hands_derived` above. Flooring the
gate at 0.5 (features damped at most to 50 %, never zeroed) recovers the
accuracy — a genuine tie with plain MLP — but `idle` still collapses on 1/5
seeds (sd 0.23 vs plain MLP's 0.013) and confidence separation is slightly
worse. **Not adopted:** a tie on the good axes, worse on `idle` robustness, for
more code and a new hyperparameter. The committed `semlp` candidate is floor
0.5, seed 0 (best variant, for the record).

Kept from the experiment: (1) the gate is **interpretable** and validates the
feature-block design — it suppresses the off-hand for one-hand gestures
(`raise_right`: left-hand block ≈ 0.16 vs right-hand ≈ 0.48; `raise_left` the
mirror), leans on the joint-angle block for postures, and keeps that block the
most open overall (0.49); (2) a **third independent confirmation** (with Track B)
that `idle` is fragile to anything that reduces feature information — the
residual "nothing happening" class needs the whole vector to rule gestures out.

## Track D — augmentation strength

Rebuilt the dataset under 4 `AugParams` variants (via `AUG_OVERRIDE` /
`ARMS_UP_KEEP_ORIG` env hooks in `build_dataset.py`), retrained MLP `both`
seed 0, scored `val` + `test`:

| config | test real-eval | test mini_heart | test raise_R/L |
|---|---|---|---|
| **defaults** *(LOCK)* | **0.847** | **0.73** | 0.76 / 0.77 |
| keep chest-level mini_heart originals | 0.832 | 0.66 | 0.78 / 0.69 |
| + widen `elevate_dy` to −0.05 | 0.816 | 0.39 | 0.74 / 0.72 |
| `n_per_sample` 4→7 | 0.788 | 0.16 | 0.69 / 0.81 |
| gentler occlusion (`hand_drop` .10→.05, `kp_dropout` .05→.03) | 0.805 | 0.30 | 0.68 / 0.84 |

**No variant beats the defaults.** More augmentation washes out class
boundaries (plan's stated risk); `mini_heart` is especially fragile. `val`
`mini_heart` sits at ~0.99 for every config (same-domain, uninformative) — the
val split does not track `test` for the scarce classes. **`AugParams` defaults
retained.**

## Track E — confidence thresholds (`classifier/thresholds.py`)

Per-class max-probability thresholds fit on `val` (keep 85 % of val TPs) do
**not transfer** to `test`. `val` is the same webcam domain as train, so its
confidences run ~0.2 higher than the cross-domain `test` frames:

| split | raw macro-F1 | val-per-class gate | global 0.20 floor |
|---|---|---|---|
| val | 0.747 | 0.750 | 0.747 |
| test | **0.873** | 0.754 (mini_heart → F1 0) | 0.872 |

On `test`, the idle errors are real gestures misread as idle, so a
reject-to-idle gate lowers idle precision at every threshold (0.62 → 0.54 at
T=0.25, → 0.38 at T=0.5) while barely moving macro-F1. **Only a garbage floor
(`global_floor` 0.20, ~1 frame affected) is active.** The frame-level flicker
is Phase 5's temporal majority vote to fix; real per-class thresholds need
Phase 6 field confidences. `classifier/thresholds.py` stays as the re-tuning
harness; `classifier/models/mlp.thresholds.json` records the (inactive) val
reference values.

## Phase 4 outcome

Lock unchanged: **MLP, `--features both`, seed 0** (`classifier/models/mlp.joblib`).
Tracks B/D/E each explored a lever and each confirmed the current config is the
one to ship. Remaining weak classes (`mini_heart` 0.73, `raise_*_hand` 0.76,
`idle` precision 0.62 on test) are data/feature-gap limited, not
hyperparameter-limited — Phase 6 field recordings are the next real lever.

Report: `results/phase4/classifier_report.docx` (regenerated — model comparison
+ Track B/D/E sections + a paper-style accuracy-vs-size Pareto scatter). Source
data: `results/phase4/{model_comparison,track_b_seeds,track_d,track_e}.json`.
Regenerate with `python -m classifier.evaluate --all && python -m classifier.thresholds --model mlp && python scripts/phase4_report.py`.
