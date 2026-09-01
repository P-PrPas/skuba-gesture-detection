# Phase 3 — baseline classifier

`classifier/train.py` (LightGBM + RandomForest), `classifier/evaluate.py`,
`features/derived.py`. Full report: `results/phase3/classifier_report.docx`.

Train: `data/dataset/train.npz` (~57k rows — external features + augmentation).
Test: `data/dataset/test.npz` (1,581 rows — 1,367 original s01/s02 frames + 214
held-out HaGRID `thumb`, never augmented). Metrics are computed only over the
**13 modelled classes'** frames.

## Features

`--features both` (default): the 152-d normalised keypoints + 37 derived
features (joint angles + length ratios + inter-wrist distance,
`features/derived.py`). The derived features are scale-invariant — MediaPipe
Hands on a tight robot-camera wrist-crop produces a differently proportioned
skeleton than on a large webcam hand, which is what broke `rock` in the first
pass.

## 2 classes are not modelled (was 3 — `mini_heart` is now fixed)

### `i_love_you` — reproduced backbone-resolution failure (CLAUDE.md #4), not a data gap

We pulled **606 real ILY images** from 5 Roboflow Universe ASL datasets
(`roboflow_ily` source) and trained on them. Result: `i_love_you` recall stayed
**0.00** (all 141 s01 test frames → `rock`) *and* it dragged `rock` 0.78 → 0.48.
Identical outcome to the earlier aug(s01) attempt.

Root cause, measured on s01 hand features (`features/derived._hand_derived`):
MediaPipe Hands on s01's wrist-crop reports **every finger curled** for
`i_love_you`, `rock` and `two_finger` alike — `idxCurl` ≈ 2.7 rad (a straight
finger is ~0.4), `thumbCurl` ≈ 2.85, `thumbPinkyAng` 0.26 vs 0.31. The three
classes are **the same feature vector**. No classifier-layer or training-data
fix separates identical inputs.

`rock` / `two_finger` only "work" because HaGRID teaches those labels for
"loose-fist-shaped hand + upright body" and s01's executions happen to match;
`i_love_you` has no such luck and collapses into `rock`.

**Crop reframing does NOT fix it (tested).** A two-stage crop
(`HandLandmarker.detect_at_wrist`, prototyped then reverted) — loose crop to
locate the hand, then a tight re-crop with the hand filling ~65% of the frame —
left the features unchanged: ILY `thumbPinkyAng` 0.26 vs rock 0.28,
`idxCurl` 2.80 vs 2.93. (An earlier wrist-centred 0.3×-shoulder crop *seemed* to
separate them, but that came with a 20% detection drop from clipped hands — the
"separation" was a partial-hand artifact.) MediaPipe Hands genuinely cannot
resolve finger extension on this footage regardless of framing.

Remaining options (all out of Phase 3 scope):
1. Swap the hand-landmark model (backbone change).
2. Re-record ILY / rock / two_finger closer & sharper.
3. Accept `i_love_you` / `rock` / `two_finger` as a known confusion cluster and
   lean on temporal + context cues in Phase 5, or collect them in Phase 6.

The 606 ILY feature rows are kept in `data/features_ext/roboflow_ily__*.npz` for
whichever fix lands; `i_love_you` stays in `PENDING_DATA`.

### `heart` — no data

No dataset for the overhead 2-arm heart. The `_classify_coco` heart branch
(wired, `docs/external_datasets.md` R2.3) finds only **55 candidates** in all of
COCO-train, mostly false positives (reaching / diving poses that match "wrists
above eyes, together, elbows out"). MediaPipe would keep ~half → too few, and
the earlier 48/606-row `i_love_you` runs showed a class this small poisons its
neighbours. Not extracted. Phase 6 field data.

Both stay in the class list and `test.npz`, excluded from training, flagged
`pending_data`. Test frames land as: `i_love_you` → rock, `heart` → mini_heart.
Data-side history for both is in `results/phase2/dataset_report.docx` §7-8.

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
| RandomForest | **0.87** | **0.82** | 0.63 | ~300 MB |
| LightGBM | not re-run (laptop RAM) — Phase 4 on Colab | | | 30 MB |

Per class (RF): squat/laying/glico 1.00 (leak), thumb 0.97, **mini_heart 0.93**
(prec 1.00 / rec 0.87), raise_left 0.90, t_pose 0.86, raise_right 0.82
(prec 0.99), rock 0.78, ok 0.78, two_finger 0.72, sit 0.97 (leak; s02 0.63),
idle 0.62 (precision 0.45 — over-triggers, recall 1.0).

Augmentation seeds are keyed on the class index (`build_dataset.py`), so a
rebuild is now reproducible — the old `hash(cls)` seed varied per process and
these numbers drifted ~±0.03 between builds.

RandomForest is the working default: from the first (12-class) pass `sit`@s02
was 0.68 (RF) vs 0.45 (LightGBM) — the only honest cross-person number — and RF
confidences are spread (LightGBM outputs ~1.0 on everything, which would break
the Phase 5 idle threshold). LightGBM was not re-run for the 13-class pass (the
dev laptop is at ~0.45 GB free RAM); the head-to-head re-runs on Colab in
Phase 4, which is where the model lock happens anyway.

## Phase 3 status: usable baseline, pipeline ready for Phase 4

13/15 classes at macro-F1 ~0.86, no class below 0.6. The train/eval pipeline
(features → model → per-class + cross-domain + cross-person eval) is what
Phase 4 reuses for the MLP and the feature comparison.

Carried into Phase 4/5:
- LightGBM vs RF head-to-head on Colab; add the MLP; lock one.
- per-class confidence thresholds + the idle/unknown fallback.
- re-extract COCO with a person-bbox crop (COCO gives the bbox) to lift
  raise_hand / t_pose recall (now ~0.75).
- `rock` / `ok` / `two_finger` sit at 0.72-0.78 — MediaPipe Hands is near its
  ceiling on this footage (same limit that sinks `i_love_you`); a hand-model
  swap is the only lever left, evaluate in Phase 4.
- `i_love_you` / `heart` — Phase 6 field data (or a hand-model swap / re-record).
