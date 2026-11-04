# Training data sources

The two clips in `dataset/` (IMG_8242, IMG_8243) are the **held-out test set** and are never used for training
or threshold selection. Everything else comes from open datasets, downloaded by `scripts/fetch_datasets.py`.

| Source | What we took | Size on disk | How it is used | License / caveat |
|---|---|---|---|---|
| **Kinetics-700** (HF mirror, read by HTTP range from the zips) | class `waving hand` (400 clips) as positives; 40 "hard-negative" classes x50 (adjusting glasses, fixing hair, scrubbing face, yawning, phone, drinking, eating, smoking, ...); 13 "neutral" classes x50 (watching tv, reading, writing, ...); `saluting` kept but excluded | 3.1k clips, 3.3 GB | frame labels from pose (see below) | YouTube videos; the HF card says license "other" - **not verified**, treat as research use only. `waving hand` is dominated by babies/children. |
| **HMDB51** (HF mirror) | `wave` (104) positive; 944 hard-negative (drink, eat, smoke, brush_hair, chew, smile, laugh, talk) and 1,084 neutral (walk, turn, stand, sit) clips | 2,132 clips, 289 MB | same | CC BY 4.0 (Serre lab). Low resolution, movie clips. |
| **SCB-Dataset, Handrise-Read-Write** (HF `wintonYF/SCB-Dataset`) | 6,864 classroom images with per-person boxes: hand-raising / reading / writing | 744 MB | seated people, the closest domain to the robot; reading+writing are hard negatives | non-commercial / all rights reserved by the authors; students (children), classroom camera |

Blocked / not used: Roboflow Universe (needs API key), AVA, NTU RGB+D (registration), Baidu-hosted sets.

## Labels (what is weak, and how it is cleaned)

Kinetics/HMDB labels are per **clip**, not per frame. `pipeline/build_dataset.py` converts them to frame labels:

* `waving hand` / `wave` clips: a frame is positive only if a wrist is >0.10 shoulder-widths above its shoulder;
  other frames of that clip are **dropped** (unknown), not labelled negative.
* hard-negative / neutral clips: label 0, except frames whose wrist is >0.55 above the nose (clearly above the
  head - may be a genuine raise) which are dropped.
* SCB: boxes matched to YOLO-pose persons by IoU>=0.5; `hand-raising` -> 1 (if a wrist is above a shoulder),
  `reading`/`writing` -> 0; unmatched persons dropped.
* frames without both shoulders, or with a person box < 20 % of frame height (SCB: <100 px), are dropped.

This is **label cleaning** with geometry, so the classifier inherits the "wrist above shoulder" definition for
positives. What it learns beyond the rule comes from the negatives (face/head-touching, phone, drinking, ...).

## Known gaps

* No dataset has a real "scratch head" class; the closest are fixing/brushing hair, scrubbing face, adjusting
  glasses, phone. **Record in-house hard negatives** (scratch head, adjust glasses, drink, phone, stretch,
  hand on chin) with the robot camera - this will matter more than model size.
* Positives are mostly children (K700) or classroom students (SCB); few seated adults on a robot camera.
* Test set is two clips of one person: it checks the pipeline, it cannot support claims about generalisation.
