# Phase 3 — baseline classifier

`classifier/train.py` (LightGBM + RandomForest), `classifier/evaluate.py`,
`features/derived.py`. Full report: `results/phase3/classifier_report.docx`.

Train: `data/dataset/train.npz` (~51k rows — external features + augmentation).
Test: `data/dataset/test.npz` (1,581 rows — 1,367 original s01/s02 frames + 214
held-out HaGRID `thumb`, never augmented). Metrics are computed only over the
**12 modelled classes'** frames.

## Features

`--features both` (default): the 152-d normalised keypoints + 36 derived
features (joint angles + length ratios, `features/derived.py`). The derived
features are scale-invariant — MediaPipe Hands on a tight robot-camera wrist-crop
produces a differently proportioned skeleton than on a large webcam hand, which
is what broke `rock` in the first pass.

## 3 classes are not modelled

`i_love_you`, `heart`, `mini_heart` have **no viable training path**:
- ASL "I love you" handshape is in no public static-image dataset.
- the overhead 2-arm `heart` is in no dataset.
- HaGRID's `hand_heart` is a chest-level finger-heart; s01's `mini_heart` is
  hands-together overhead — it does not transfer.

Training them on augmented s01 frames dragged `rock` to 0.00 (s01's `rock` and
`i_love_you` were performed too similarly in one session). They stay in the
class list and in `test.npz`, excluded from training, flagged `pending_data`.
Their test frames land as: i_love_you -> rock, heart/mini_heart -> raise_right_hand.

## Results (12 modelled classes)

| model | macro-F1 (12) | macro-F1 (8 real-eval) | sit@s02 (clean cross-person) | size |
|---|---|---|---|---|
| LightGBM | **0.88** | **0.83** | 0.45 | 30 MB |
| RandomForest | 0.87 | 0.81 | **0.68** | 306 MB |

Per class (RF): squat/laying/glico 1.00 (leak), thumb 0.97, raise_left 0.90,
t_pose 0.88, raise_right 0.82, ok 0.78, rock 0.78, two_finger 0.74, sit 0.97
(leak; s02 0.68), idle 0.64 (precision 0.47 — over-triggers, recall 1.0).

RandomForest is the working default: `sit`@s02 is the only honest cross-person
number and RF gets 0.68 vs LightGBM 0.45, and RF confidences are spread
(LightGBM outputs ~1.0 on everything, which would break the Phase 5 idle
threshold). Final lock is a Phase 4 decision.

## Phase 3 status: usable baseline, pipeline ready for Phase 4

12/15 classes at macro-F1 ~0.85, no class below 0.6, one clean cross-person
number (sit@s02 0.68). The train/eval pipeline (features -> model -> per-class
+ cross-domain + cross-person eval) is what Phase 4 reuses for the MLP and the
feature comparison.

Carried into Phase 4/5:
- per-class confidence thresholds + the idle/unknown fallback.
- re-extract COCO with a person-bbox crop (COCO gives the bbox) to lift
  raise_hand / t_pose recall (now ~0.72).
- `i_love_you` / `heart` / `mini_heart` — only with new recordings or Phase 6
  field data.
