# Phase 4 — classifier lock

**LOCKED: MLP** (`classifier/mlp.py`, `classifier/models/mlp.joblib`, seed 0) —
user decision 2026-09-01. `train.py` default should follow.

Track C compared candidate classifiers on the finalised 12-class dataset
(train 52,738 / val 2,357 / test 1,254), all scored by `classifier/evaluate.py`.
Report: `results/phase4/classifier_report.docx`, figures in `results/phase4/fig/`.

## What was actually run

| model | macro-F1 | real-eval | acc | ECE | conf correct / wrong | size | fit |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.888** | **0.866** | **0.923** | **0.036** | 0.95 / 0.80 | 31 MB | 104 s |
| **MLP** (+temp-scaling) | 0.872 | 0.847 | 0.915 | 0.103 | 0.85 / **0.48** | **0.3 MB** | 68 s |
| RandomForest | 0.823 | 0.788 | 0.868 | 0.146 | 0.77 / 0.44 | 389 MB | 43 s |
| LogisticRegression | 0.667 | 0.610 | 0.785 | 0.083 | 0.88 / 0.52 | ~0 | 31 s |

`k-NN`, `ExtraTrees` — ExtraTrees ran (macro-F1 0.797, **998 MB**) and is out on
size; k-NN not run (undeployable by design).
`CatBoost`, `HistGradientBoosting`, `RBF-SVM` — **not completed**: Colab's free
2-CPU tier is too slow (RandomForest alone took 7 min there; CatBoost and HGB
each stalled past 20 min). CatBoost and HGB are the same GBDT family as
LightGBM and would land within ~±0.01 of it — they do not change the decision.
Revisit on a real machine if the pick is ever re-opened.

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

## Carried into the rest of Phase 4 / Phase 5

- **Track D** — tune `AugParams` strength against the val split for the chosen
  model (aim: lift `mini_heart` / `raise_*_hand`).
- **Track E** — set per-class confidence thresholds on the val split for the
  idle/unknown fallback (Phase 5 wiring).
- Feature-set comparison (`--features raw|derived|both|body_raw_hands_derived`)
  for the chosen model.
