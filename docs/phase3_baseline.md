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

## 2 classes are not modelled (was 3 — `mini_heart` is now fixed)

`i_love_you`, `heart` have no viable training path yet:
- ASL "I love you" handshape: only on Roboflow Universe (needs an API key or a
  manual export — not pulled). See `docs/external_datasets.md` R2.2.
- the overhead 2-arm `heart`: no dataset. A COCO-mining filter branch is wired
  (`_classify_coco`) but not yet extracted. R2.3.

Training them on augmented s01 frames dragged `rock` to 0.00 (s01's `rock` and
`i_love_you` were performed too similarly in one session). They stay in the
class list and in `test.npz`, excluded from training, flagged `pending_data`.
Test frames land as: `i_love_you` -> rock, `heart` -> mini_heart.

**`mini_heart` fix (Round 2, R2.4).** HaGRIDv2 `hand_heart` gives the right
handshape; the per-hand normalisation makes the hand slice position-invariant,
so only the *body* pose carried the chest-vs-overhead gap. Fix, no new data:
`features/augment.raise_arms` shifts both forearms + hand points up by a shared
`dy` (keeps the two hands together) — applied to the HaGRID rows in
`build_dataset.py` (`ARMS_UP` set). Plus one new derived feature: inter-wrist
distance in body units (`features/derived._body_derived`). Result:
`mini_heart` 0.00 -> **0.92 F1** (prec 1.00 / rec 0.84), and `raise_right_hand`
went 0.67 -> 0.82 as a bonus (the elevation aug gave the model cleaner
arms-overhead negatives).

## Results (13 modelled classes)

| model | macro-F1 (13) | macro-F1 (9 real-eval) | sit@s02 (clean cross-person) | size |
|---|---|---|---|---|
| RandomForest | 0.87 | 0.82 | 0.62 | ~300 MB |
| LightGBM | _see report_ | | | 30 MB |

Per class (RF): squat/laying/glico 1.00 (leak), thumb 0.97, **mini_heart 0.92**,
raise_left 0.89, t_pose 0.86, raise_right 0.82, rock 0.78, two_finger 0.74,
ok 0.77, sit 0.97 (leak; s02 0.62), idle 0.62 (precision 0.44 — over-triggers,
recall 1.0).

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
