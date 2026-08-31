# Phase 2b — train on external data, test on s01/s02

Decision (user, 2026-08-30, RoboCup@Home context → licences are not a blocker):
train the classifier on public datasets and keep **every original s01/s02 frame
as test-only**. Classes with no external data (`sit`, `laying`, `squat`,
`glico_pose`) are trained on **augmented s01/s02 frames only** — the originals
still go to test. `i_love_you` and `heart` have no viable source yet and are
`PENDING_DATA` (in the class list + test.npz, 0 train rows — see
`docs/external_datasets.md` Round 2). `mini_heart` was in that group through the
first Phase 3 pass; it is now trained on HaGRIDv2 `hand_heart` + an
arm-elevation augmentation (Round 2, R2.4).

Full dataset survey + licences: `docs/external_datasets.md`.

## Per-class train/test source

| class | TRAIN source | TEST source | notes |
|---|---|---|---|
| `idle` | HaGRID `no_gesture` + COCO person crops (no gesture) | s01 `idle_01` | |
| `raise_right_hand` | COCO, filtered (one wrist above nose, other down) | s01 | heuristic-labelled, spot-checked |
| `raise_left_hand` | mirror-aug of `raise_right_hand` (schema `MIRROR_LABEL_SWAP`) | s01 | no own external data needed |
| `sit` | **aug(s01 `sit_01/02`) only** — no clean external source (NTU registration, Le2i link dead) | s01 `sit_01/02` + s02 `sit_03` | s02 `sit_03` is still a real cross-person test |
| `squat` | Kaggle gym-workout `squat` clips (mid-rep frames) — needs `kaggle.json` | s01 `squat_01` | falls back to aug(s01) if the pull is blocked |
| `laying` | **aug(s01... none) → aug(s02 `laying_03`) only** | s02 `laying_03` (originals) | s01 has no usable laying; joins the aug-only group |
| `ok` | HaGRID `ok` | s01 `ok_01` | |
| `i_love_you` | **not modelled yet** — `PENDING_DATA`. ASL ILY handshape is only on Roboflow Universe (needs an API key / manual export, not pulled). HaGRID `call` wired as hard negative. | s01 `i_love_you_01` (originals) | see `docs/external_datasets.md` R2.2 |
| `rock` | HaGRID `rock` | s01 `rock_01` | |
| `two_finger` | HaGRID `peace` (+ `peace_inverted`, `two_up`) | s01 `two_finger_01` | |
| `thumb` | HaGRID `like` | **held-out HaGRID subjects** (s01 has zero thumb) | tag HaGRID rows by worker id, hold ~15% of subjects out |
| `heart` | **not modelled yet** — `PENDING_DATA`. No dataset for the overhead 2-arm heart; a COCO-mining filter branch is wired (`_classify_coco`) but not yet extracted. | s01 `heart_01/02` (originals) | see R2.3 |
| `mini_heart` | HaGRID v2 `hand_heart` + `hand_heart2` **+ arm-elevation augmentation** (`features/augment.raise_arms`) — HaGRID is chest-level, s01 is overhead; the per-hand normalisation makes the handshape position-invariant so only the body needs lifting | s01 `mini_heart_01` | see R2.4 |
| `t_pose` | COCO, filtered (both wrists ≈ shoulder height, arms extended, standing) | s01 `t_pose_01` | heuristic-labelled, spot-checked |
| `glico_pose` | **aug(s01 `glico_pose_01`) only** | s01 `glico_pose_01` (originals) | no dataset |

## What each test number means

- **External-trained classes** (`ok`, `rock`, `two_finger`, `mini_heart`, `idle`,
  `sit`, `squat`, `laying`, `raise_*_hand`, `t_pose`): s01/s02 is a genuine
  held-out cross-subject **and** cross-domain test. This is a real generalisation
  number.
- **`thumb`**: cross-subject within HaGRID (held-out workers). Real, but same
  domain (webcam).
- **`i_love_you`, `heart`, `glico_pose`**: train = aug(same frames), test =
  originals. This is **not** a generalisation number — it measures "does the
  model recognise this pose when it has seen many geometric variations of these
  exact instances". Report it, always with this caveat. Phase 6 field testing is
  the real gate for these three.

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
