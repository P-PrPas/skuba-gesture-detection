# Phase 3 — baseline classifier

`classifier/train.py` (LightGBM + RandomForest), `classifier/evaluate.py`,
`features/derived.py`. Full report: `results/phase3/classifier_report.docx`.

**Status: DONE (2026-09-01).** RandomForest, `--features both`, **12 classes**,
macro-F1 0.88 / 0.82 over the 8 real-generalisation classes, every class ≥ 0.6.

Train: `data/dataset/train.npz` (~50k rows — external features + augmentation).
Test: `data/dataset/test.npz` (~1,250 rows — original s01/s02 frames + 214
held-out HaGRID `thumb`, never augmented). Metrics are over the modelled classes.

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
| macro-F1 (12) | **0.88** |
| macro-F1 (8 real-eval: cross_domain + held_out_external) | **0.82** |
| sit @ s02 (clean cross-person) | 0.53 (60 frames — noisy; still aug-only) |

Per class: squat/laying/glico 1.00 (leak), **thumb 0.99**, **two_finger 0.94**,
raise_left 0.90, t_pose 0.84, raise_right 0.77, ok 0.78, mini_heart 0.75,
sit 0.96 (leak; s02 0.53), idle 0.61 (precision 0.43 — over-triggers, recall 1.0).

Augmentation seeds are keyed on the class index (`build_dataset.py`) — a rebuild
is reproducible (the old `hash(cls)` seed varied per process, ±0.03 drift).

LightGBM was not re-run for the 12-class pass (dev laptop at ~0.45 GB free RAM).
From the first 12-class pass RF's `sit`@s02 was 0.68 vs LightGBM 0.45 and RF
confidences are spread (LightGBM pins ~1.0 on everything, which breaks the
Phase 5 idle threshold). RF is the working default; the head-to-head + the
model lock happen in Phase 4 on Colab.

## Phase 3 status: usable baseline, pipeline ready for Phase 4

12/12 classes at macro-F1 0.88, every class ≥ 0.6. The train/eval pipeline
(features → model → per-class + cross-domain + cross-person eval) is what
Phase 4 reuses.

Carried into Phase 4/5:
- LightGBM vs RF head-to-head on Colab; add the MLP; lock one.
- **de-leak sit/squat/laying** — COCO `_classify_coco` now has posture branches;
  extract on Colab so they become `cross_domain` instead of `aug_only_leak`.
- per-class confidence thresholds + the idle/unknown fallback (idle over-triggers,
  precision 0.43 — several gestures fall to it at ~0.7 recall).
- **evaluate a hand-landmark model swap** — `ok` / `two_finger` and the cut
  `i_love_you` / `rock` are all limited by MediaPipe Hands' finger resolution on
  distant footage; this is the biggest lever left.
- re-extract COCO with a person-bbox crop to lift raise_hand / t_pose recall.
