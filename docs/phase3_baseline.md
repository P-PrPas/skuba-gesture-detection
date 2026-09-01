# Phase 3 — baseline classifier

`classifier/train.py` (LightGBM + RandomForest), `classifier/evaluate.py`
(reusable eval harness — JSON + figures + `--all` comparison), `features/derived.py`.
Full report + figures: `results/phase3/classifier_report.docx`, `results/phase3/fig/`.

## LightGBM vs RandomForest (12 classes, features=both)

| metric | RandomForest | LightGBM |
|---|---|---|
| macro-F1 (12) | **0.84** | 0.83 |
| macro-F1 (10 real-eval) | **0.81** | 0.80 |
| accuracy | **0.88** | 0.87 |
| sit @ s02 (cross-person) | 0.97 | **1.00** |
| laying | 0.71 | **0.88** |
| mini_heart | **0.75** | 0.57 |
| conf when correct / wrong | 0.77 / 0.46 | 0.96 / 0.72 |
| ECE (calibration) | 0.148 | **0.061** |
| model size | 419 MB | **31 MB** |
| fit time | **48 s** | 327 s |

**RandomForest is the working default** — its confidence separates right from
wrong (0.77 vs 0.46, spread histogram), which is what the Phase 5 idle/unknown
threshold needs. LightGBM piles ~80% of predictions at confidence ~1.0 (0.96
correct / 0.72 wrong) — aggregate-calibrated but not thresholdable without Platt/
isotonic calibration. LightGBM's 31 MB (vs RF's alarming 419 MB) and better
`laying` keep it a Phase 4 candidate. Model lock (incl. an MLP) is a Phase 4
decision, made with this same harness.

**Status: DONE (2026-09-01).** RandomForest, `--features both`, **12 classes**,
macro-F1 0.84 / **0.81 over the 10 real-generalisation classes**, every class ≥ 0.6.

Train: `data/dataset/train.npz` (~55k rows — external features + augmentation).
Test: `data/dataset/test.npz` (1,254 rows — original s01/s02 frames + 214
held-out HaGRID `thumb`, never augmented). Metrics are over the modelled classes.

**`sit` and `laying` were de-leaked from COCO** — 10/12 classes now have a real
cross-domain test number (was 8). `sit` cross-person recall jumped 0.53 → **0.97**.

## Features

`--features both` (default): 152-d normalised keypoints + 37 derived features
(joint angles + length ratios + inter-wrist distance, `features/derived.py`).
Scale-invariant — MediaPipe Hands on a tight robot-camera wrist-crop produces a
differently proportioned skeleton than on a large webcam hand.

## 3 classes were cut (15 → 12)

`i_love_you`, `rock`, `heart` — removed from `features/schema.py`. Full data
history: `results/phase2/dataset_report.docx` §7-8.
- `i_love_you` / `rock`: MediaPipe Hands reports "all fingers curled" for
  `i_love_you` / `rock` / `two_finger` alike on s01's conversational-distance
  footage — same feature vector. Tried: aug(s01), 606 real Roboflow ILY images,
  a two-stage tight crop. None separated them. `two_finger` is kept — HaGRID
  anchors "loose fist + upright body" to that label and it still works.
  Revisit `i_love_you`/`rock` if a better hand model is adopted (Phase 4) — the
  606 ILY feature rows are kept at `data/features_ext/roboflow_ily__*.npz`.
- `heart`: no dataset; the COCO-mining filter finds only 55 noisy candidates.

Cutting `rock` **fixed `two_finger`**: 0.72 → 0.94 (it was mostly confused with
`rock`). `thumb` also rose 0.97 → 0.99.

## `mini_heart` — modelled via augmentation, no new data

HaGRIDv2 `hand_heart` is a chest-level finger-heart; s01's `mini_heart` is
hands-together overhead. The per-hand normalisation makes the handshape slice
position-invariant, so only the body pose carried the gap.
`features/augment.raise_arms` shifts both forearms + hand points up by a shared
`dy` (keeps the hands together), applied to the HaGRID rows in `build_dataset.py`
(`ARMS_UP`). Plus an inter-wrist-distance derived feature. `mini_heart` went
0.00 → 0.75-0.93 F1 (45-frame test, run-to-run variance is wide — watch in
Phase 4).

## Results (12 modelled classes, RF, features=both)

| metric | value |
|---|---|
| macro-F1 (12) | **0.84** |
| macro-F1 (10 real-eval: cross_domain + held_out_external) | **0.81** |
| **sit @ s02** (clean cross-person) | **0.97** (was 0.53 when aug-only) |

Per class: **thumb 1.00**, **two_finger 0.97**, glico 1.00 / squat 1.00 (leak —
still aug-only), **sit 0.91** (s02 0.97), t_pose 0.84, ok 0.78, raise_right 0.77,
raise_left 0.76, mini_heart 0.75 (45-frame test), **laying 0.71** (prec 1.00 /
rec 0.55 — honest cross-domain now, was a leaked 1.00), idle 0.63 (precision
0.46 — over-triggers, recall 0.98).

### sit / squat / laying — COCO posture mine

`_classify_coco` gained sit/squat/laying branches (spine-horizontal → laying;
knee-bend + hip-vs-knee height → sit/squat), re-verified in `_clean_external`.
- **sit**: 1,372 clean COCO rows. sit went from a leaked 0.96 / real-0.53 to an
  honest **0.91 F1, 0.97 cross-person**. The best single improvement in Phase 3.
- **laying**: 607 clean COCO rows. Honest 0.71 (was a leaked 1.00). Misses come
  from `laying → sit` / `laying → raise_left_hand` — a lying person's arm reads
  as raised in un-rotated normalized coords. Phase 4/5 item.
- **squat**: COCO's "squat" auto-labels are shallow crouches (mean knee angle
  107°, not a deep squat) and overlapped `sit` badly — with COCO squat in,
  `squat` F1 collapsed to 0.01 and `sit` fell to 0.58. Reverted: `squat` stays
  aug(s01)-only (leaked 1.00). Needs a real squat dataset (NTU / gym) — Phase 6.

Augmentation seeds are keyed on the class index (`build_dataset.py`) — a rebuild
is reproducible (the old `hash(cls)` seed varied per process, ±0.03 drift).

## The eval harness

`classifier/evaluate.py` scores any bundle `{clf, classes, clip, features}` whose
`clf` has `.predict_proba` + `.classes_`. `--model <m>` writes `<m>.eval.json`
(per-class P/R/F1, macro variants, accuracy, balanced-acc, confusion matrix,
cross-person recall, confidence-when-correct-vs-wrong, 10-bin reliability + ECE,
size, fit time) plus `<m>_{confusion,perclass,calibration}.png`. `--all` adds
`model_comparison.json` + `compare_{f1,overall}.png`. Every future candidate (MLP,
another tree lib) plugs into the same bundle shape and is scored identically.

## Phase 3 status: usable baseline, pipeline ready for Phase 4

12/12 classes at macro-F1 0.84 (RF) / 0.83 (LGBM), every class ≥ 0.6. The
train/eval pipeline (features → model → harness) is what Phase 4 reuses.

Carried into Phase 4/5:
- LightGBM vs RF head-to-head on Colab; add the MLP; lock one.
- **`laying` 0.71** — tighten the COCO filter and/or add a "torso horizontal"
  emphasis so a lying arm stops reading as `raise_left_hand`.
- **`squat`** — find a real squat source (NTU A-something, a gym dataset); COCO
  has none.
- per-class confidence thresholds + the idle/unknown fallback (idle over-triggers,
  precision 0.46 — several gestures fall to it at ~0.7 recall).
- **evaluate a hand-landmark model swap** — `ok` / `two_finger` and the cut
  `i_love_you` / `rock` are all limited by MediaPipe Hands' finger resolution on
  distant footage; biggest lever left.
- re-extract COCO with a person-bbox crop to lift raise_hand / t_pose recall.
