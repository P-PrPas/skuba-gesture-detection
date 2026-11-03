# Phase 2b — train on external data, test on s01/s02

Decision (user, 2026-08-30, RoboCup@Home context → licences are not a blocker):
train the classifier on public datasets and keep **every original s01/s02 frame
as test-only**.

**Vocabulary now 12 classes** — `i_love_you`, `rock`, `heart` were cut in
Phase 3 (2026-09-01), see `results/phase2/dataset_report.docx` §7-8 and
`docs/phase3_baseline.md`. `glico_pose` is trained on augmented s01 frames only.
`sit`/`squat`/`laying` were aug-only and are being de-leaked from COCO
(`_classify_coco` posture branches — extract on Colab). `mini_heart` is trained
on HaGRIDv2 `hand_heart` + an arm-elevation augmentation (Round 2, R2.4).

Full dataset survey + licences: `docs/external_datasets.md`.

## Per-class train/test source

| class | TRAIN source | TEST source | notes |
|---|---|---|---|
| `idle` | HaGRID `no_gesture` + COCO person crops (no gesture) | s01 `idle_01` | |
| `raise_right_hand` | COCO, filtered (one wrist above nose, other down) | s01 | heuristic-labelled, spot-checked |
| `raise_left_hand` | mirror-aug of `raise_right_hand` (schema `MIRROR_LABEL_SWAP`) | s01 | no own external data needed |
| `sit` | **COCO** `_classify_coco` posture filter (spot-checked), else aug(s01/s02) | s01 `sit_01/02` + s02 `sit_03` | s02 `sit_03` is the cross-person test |
| `squat` | **COCO** posture filter (low yield — COCO is photos), else aug(s01) | s01 `squat_01` | |
| `laying` | **COCO** posture filter (spine horizontal), else aug(s02 `laying_03`) | s02 `laying_03` | s01 has no laying |
| `ok` | HaGRID `ok` | s01 `ok_01` | |
| `two_finger` | HaGRID `peace` (+ `peace_inverted`, `two_up`) | s01 `two_finger_01` | 0.94 F1 after `rock` was cut |
| `thumb` | HaGRID `like` | **held-out HaGRID subjects** (s01 has zero thumb) | ~15% of rows held out |
| `mini_heart` | HaGRID v2 `hand_heart` + `hand_heart2` **+ arm-elevation augmentation** (`features/augment.raise_arms`) | s01 `mini_heart_01` | R2.4 |
| `t_pose` | COCO, filtered (both wrists ≈ shoulder height, arms extended, standing) | s01 `t_pose_01` | heuristic-labelled, spot-checked |
| `glico_pose` | **aug(s01 `glico_pose_01`) only** | s01 `glico_pose_01` (originals) | no dataset |

## What each test number means

- **External-trained classes** (`ok`, `two_finger`, `mini_heart`, `idle`,
  `raise_*_hand`, `t_pose`, and `sit`/`squat`/`laying` once the COCO mine
  lands): s01/s02 is a genuine held-out cross-subject **and** cross-domain test.
  A real generalisation number.
- **`thumb`**: cross-subject within HaGRID (held-out workers). Real, same domain.
- **`glico_pose`** (and `sit`/`squat`/`laying` until COCO lands): train =
  aug(same frames), test = originals. **Not** a generalisation number. Phase 6
  field testing is the real gate.

## Extraction — the streaming constraint

The dev box has ~5 GB free. **Do not** download whole datasets. `data/extract_external.py`:
1. pull one archive / class shard / video,
2. run `backbone/pose.py` + `backbone/hands.py` (+ wrist-anchored crops) on it,
3. write normalised 152-d rows to `data/features_ext/<source>__<class>.npz`
   with `source`, `subject` (dataset-native id or `<source>_<n>`), `class` tags,
4. **delete the raw files**, next shard.

Feature vectors are tiny (152 f32 ≈ 0.6 KB/frame); even 300 k frames ≈ 180 MB.
Keypoints only — no images retained.

## Pull order

1. **HaGRID v1** 512p — `ok`, `peace`, `rock`, `like`, `no_gesture` (+ `call`,
   `peace_inverted`, `two_up`). Per-class archives, streamed.
2. **HaGRID v2** 512p — `hand_heart`, `hand_heart2`.
3. **COCO 2017** (val + a train slice) — `idle` negs, `t_pose`, `raise_right_hand`.
   Filter with COCO's own 17-kpt annotations first, then MediaPipe on survivors.
4. **Le2i Fall Detection** — `sit`, `laying` (plain RGB video, lightest).
5. **NTU RGB+D** A008/A043 — more subjects for `sit`/`laying`. *Needs the user to
   register + accept the ROSE agreement; large download.*
6. **Kaggle gym-workout video** — `squat`. *Needs `~/.kaggle/kaggle.json`.*

Steps 5–6 are blocked on the user (registration / API creds / more disk). Until
then `sit`/`laying` rely on Le2i and `squat` falls back to aug(s01).

## build_dataset changes (after extraction)

`pipeline/build_dataset.py` merges `data/features_ext/*.npz` + `data/features/*.npz`:
- `train.npz` = external rows (augmented) + aug-only rows for the 3 stuck classes.
- `test.npz` = all original s01/s02 frames + held-out HaGRID-subject `thumb` frames.
- `s01`/`s02` frames NEVER appear in `train.npz` except as augmented copies of the
  3 stuck classes.
- dataset card records per-class train source, test source, and the caveat flag.
