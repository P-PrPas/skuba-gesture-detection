# Implementation plan — gesture/posture recognition

Phased roadmap. Each phase has a goal, concrete tasks, a deliverable, and an exit criterion — don't move to the next phase until the exit criterion is met.

## Phase 0 — Environment & scaffolding

**Goal:** repo builds and runs the two candidate backbones end to end on a webcam/sample video.

- Set up repo structure (see CLAUDE.md)
- Install and smoke-test candidate pose backbones (e.g. YOLO-pose, MediaPipe Pose) and MediaPipe Hands
- Confirm target deployment hardware (Jetson model / onboard PC / RGB vs RGB-D camera) — this affects Phase 1 choices, get it confirmed before proceeding

**Deliverable:** a script that reads a video/webcam stream and overlays body + hand keypoints.

**Exit criterion:** keypoints visibly track a person in a live/test feed without crashing.

## Phase 1 — Backbone sanity check (not a full benchmark)

**Goal:** pick one body-pose backbone and confirm MediaPipe Hands is adequate, based on the project's actual hard cases — not a generic accuracy comparison.

- Collect a small set of hard-case clips: person laying down (from robot's camera height), person doing rock/fist-like hand shapes with fingers overlapping, person at typical robot-interaction distance (2-4m)
- Run each candidate body-pose backbone on the laying/squat clips — check visually whether keypoints stay stable and anatomically plausible
- Run MediaPipe Hands on the fist/rock clips at typical distance and on wrist-anchored crops (see ARCHITECTURE.md) — check landmark stability
- Benchmark inference latency of the chosen combination on target hardware

**Deliverable:** ARCHITECTURE.md's "Tech stack" section filled in with the chosen backbone(s) and the evidence for the choice.

**Exit criterion:** chosen backbone(s) handle all identified hard cases acceptably, and combined latency fits the robot's real-time budget. Once this is signed off, the backbone is not revisited except for a proven, reproduced failure.

## Phase 2 — Data collection pipeline

**Goal:** a repeatable way to go from "record a clip" to "labeled, normalized feature vectors" with minimal manual effort.

- Define recording protocol (see DATA_COLLECTION_SPEC.md): multiple subjects, angles, distances, lighting conditions per class, including `idle`
- Build the extraction script: clip -> per-frame keypoints (body + both hand crops) -> normalized feature vector -> labeled record
- Build the augmentation step: mirror+relabel, rotation jitter, keypoint dropout, coordinate noise (see ARCHITECTURE.md for parameters)
- Decide storage format (e.g. parquet/JSON per clip, indexed by subject and session for correct splitting)

**Deliverable:** running the pipeline on N recorded clips produces a versioned, augmented training dataset.

**Exit criterion:** dataset covers all initial classes with multiple subjects each, split by subject (not frame) into train/val/test.

## Phase 3 — Baseline classifier

**Goal:** first working classifier, correctness over performance.

- Train LightGBM/Random Forest baseline on normalized+fused features
- Evaluate per-class precision/recall and full confusion matrix — pay special attention to confusable pairs (e.g. sit vs squat, rock vs fist/idle, ok vs thumb)
- Calibrate confidence threshold per class using the validation set (not test set)

**Deliverable:** trained baseline model + evaluation report (confusion matrix, per-class metrics, chosen thresholds).

**Exit criterion:** baseline meets a minimum bar the team agrees on before comparing alternatives — don't tune this baseline extensively yet.

**Status: DONE (2026-09-01).** RandomForest, features=both, **12 classes**
(cut i_love_you / rock / heart) — macro-F1 0.84 / **0.81 over the 10
real-generalisation classes**, every class ≥ 0.6. Report:
`results/phase3/classifier_report.docx`, `docs/phase3_baseline.md`.
- Cut `i_love_you` + `rock`: MediaPipe Hands can't tell them apart from
  `two_finger` on our footage (every path tried — aug s01, 606 real Roboflow
  ILY images, two-stage tight crop). Cutting `rock` fixed `two_finger`
  (0.72 → 0.94), `thumb` → 1.00. Data history: `dataset_report.docx` §7.
- Cut `heart`: no dataset; COCO-mining finds only 55 noisy candidates (§8).
- `mini_heart` recovered 0.00 → ~0.75 via an arm-elevation augmentation
  (`features/augment.raise_arms`) + an inter-wrist-distance feature — no new data.
- **`sit` + `laying` de-leaked from COCO** (`_classify_coco` posture branches):
  `sit` 0.53 → **0.97 cross-person**, `laying` an honest 0.71 (was a leaked
  1.00). `squat` reverted — COCO's squat labels are shallow crouches that
  overlap `sit`; stays aug(s01)-only, needs a real squat dataset.
- Per-class thresholds + idle fallback: carried into Phase 4/5.
- LightGBM vs RF head-to-head: **done** (both trained 12-class, harness-scored —
  RF 0.84 / LGBM 0.83; RF default, calibration-limited LGBM a Phase 4 candidate).

## Phase 4 — Classifier iteration

**Goal:** the actual experimentation phase — this is where most of the project's effort goes.

**Scope decision (2026-09-01):** the hand-landmark model swap and any COCO
re-extraction are **cut** — the problem classes (i_love_you, rock, heart) are
cut, time is limited, and the 12-class baseline is acceptable. Phase 4 stays on
the classification layer + the existing feature data.

### Track B — feature representation
- Compare `raw` vs `derived` vs `both` vs `body_raw_hands_derived` (the
  `--features` flag exists) for the leading model.

### Track C — model family (the core; see `docs/phase4_classifiers.md`)
- Add a **stratified `val` split** to `build_dataset.py` from the pre-augmentation
  external rows (needed for early stopping, temperature scaling, threshold
  tuning — must not touch `test.npz`).
- `train.py --model {rf, lgbm, catboost, hgb, et, svm, mlp, logreg}` — each
  emitting the same `{clf, classes, clip, features}` bundle; the eval harness
  scores them identically.
- Shortlist to actually run: **CatBoost** (ordered boosting → better calibrated
  than LGBM, compact), **HistGradientBoosting** (no new dep, free data point),
  **small regularised MLP + temperature scaling** (the plan's MLP done right,
  <1 MB), **RBF-SVM on a ~15-20k subsample** (different inductive bias, tiny
  model). Stretch: a **spatial GCN** (skeleton = graph; highest upside, real
  overfitting risk on the ~1,250-row test).
- Run **k-NN once** as a separability sanity check (not a candidate).
- **Calibration is a separable layer** — pick the model on raw accuracy + size +
  separability, then apply Platt / isotonic / temperature scaling and re-score
  ECE + conf-correct-vs-wrong.
- Shrink the RandomForest (max_depth / min_samples_leaf) — 419 MB is a
  deployment liability regardless of whether RF stays the pick.

### Track D — augmentation strength
- Tune `AugParams` against the val split — too little underfits generalization,
  too much washes out real class boundaries.

### Track E — per-class confidence thresholds (bridges to Phase 5)
- Tune a per-class threshold on the val slice for the idle/unknown fallback
  (idle currently over-triggers, precision ~0.45).

### Guardrail
- `evaluate.py` confusion matrix + per-class after **every** change; a fix for
  one class must not silently break another.

**Deliverable:** one locked classifier config + `docs/phase4_baseline.md` with
the rationale + a Phase 4 report regenerated from the harness outputs.

**Deliverable:** final chosen classifier configuration with documented rationale for why it beat the alternatives.

**Exit criterion:** acceptable accuracy on held-out subjects (not seen during training) across all classes including `idle`, with no single class catastrophically confused with another.

## Phase 5 — Temporal smoothing & real-time integration

**Goal:** stable, real-time behavior on live camera input, not just single-frame offline accuracy.

- Implement majority-vote (or similar) smoothing over a short frame window
- Wire the full pipeline (backbone -> features -> classifier -> smoothing -> thresholding) into a real-time loop
- Benchmark end-to-end latency and frame rate on target hardware

**Deliverable:** a real-time demo that classifies gestures live from camera input with visible confidence/idle output.

**Exit criterion:** stable frame rate on target hardware, and smoothing measurably reduces flicker/false triggers compared to raw per-frame output.

## Phase 6 — Field testing & iteration

**Goal:** find and close the gap between lab conditions and the robot's real operating environment.

- Test in the actual environment/lighting/camera setup the robot will use
- Log failure cases (misclassifications, missed detections) with enough context to reproduce
- Retrain only the classification layer on new data derived from failure cases — do not touch the backbone unless a failure is clearly backbone-attributable (see CLAUDE.md constraint)

**Deliverable:** a log of failure cases and the resulting dataset/model updates.

**Exit criterion:** acceptable performance in the real deployment environment across repeated test runs.

## Phase 7 — Packaging for deployment

**Goal:** a versioned, deployable artifact — the final ask of this project.

- Export final classifier weights + the exact normalization/feature-fusion code it depends on (these must ship together — a classifier without its matching feature pipeline is useless)
- Document the exact backbone version/config used, since features depend on it
- Record class list, confidence thresholds, and known limitations in a short model card

**Deliverable:** a single versioned package (weights + feature pipeline code + model card) ready to load into the robot's perception stack.

**Exit criterion:** the packaged model runs correctly in the robot's actual runtime environment, not just the development environment.
