# Running the external-data extraction off the dev laptop

The SKUBA dev laptop has ~0.5 GB free RAM and ~7 GB free disk. `pipeline/
extract_external.py` OOMs there (a HaGRID parquet shard is ~0.5 GB and MediaPipe
adds ~0.2 GB). Run it on **Google Colab** (free CPU runtime: ~12 GB RAM, ~100 GB
disk) or any box with ≥8 GB free RAM, then bring back the `.npz` feature files —
they are tiny (~0.6 KB per frame, so the whole external set is tens of MB).

Everything downstream — `pipeline/build_dataset.py`, Phase 3 training,
evaluation — is light and stays on the laptop.

## Colab cells

```python
# 1. clone + deps
!git clone https://github.com/P-PrPas/skuba-gesture-detection.git
%cd skuba-gesture-detection
!pip -q install "mediapipe==1.0.1" "opencv-contrib-python==4.11.0.86" "numpy>=1.26,<3" pyarrow requests
!pip -q install roboflow          # only for the roboflow_ily source
```

```python
# 2. run each source (resumable — re-run the same line if a shard times out)
!python -m pipeline.extract_external --list

# batch 1 — HaGRID hand gestures + idle  (~30-60 min CPU)
!python -m pipeline.extract_external hagrid
!python -m pipeline.extract_external hagrid_nogesture

# batch 2 — mini_heart + COCO poses  (~1-2 h CPU; COCO annotations are ~250 MB)
!python -m pipeline.extract_external hagrid_v2_heart
!python -m pipeline.extract_external coco_pose   # now also mines `heart` (arms-overhead filter)

# batch 3 — i_love_you from Roboflow Universe  (~10-20 min)
import os; os.environ["ROBOFLOW_API_KEY"] = "PASTE_YOUR_FREE_KEY"  # roboflow.com -> Settings -> API
!python -m pipeline.extract_external roboflow_ily
```

```python
# 3. sanity-check + zip the features to download
import numpy as np, glob
for f in sorted(glob.glob("data/features_ext/*.npz")):
    d = np.load(f)
    print(f, d["X"].shape, "finite" if np.isfinite(d["X"]).all() else "NON-FINITE",
          "subjects", len(set(d["subject_id"].tolist())))
!cd data && zip -r features_ext.zip features_ext
from google.colab import files; files.download("data/features_ext.zip")
```

## Back on the laptop

```bash
unzip ~/Downloads/features_ext.zip -d "C:/Code/SKUBA/gesture detection/data/"
git add -f data/features_ext/*.npz        # small; worth committing
python -m pipeline.build_dataset           # merges features_ext/ + features/
```

## Batches

| batch | sources | classes it covers | ~Colab time |
|---|---|---|---|
| 1 | `hagrid`, `hagrid_nogesture` | ok, two_finger, rock, thumb, idle (+ ILY hard-negative) | ~30–60 min |
| 2 | `hagrid_v2_heart`, `coco_pose` | mini_heart, t_pose, raise_right_hand, raise_left_hand, **heart** (+ more idle) | ~1–2 h |
| 3 | `roboflow_ily` | **i_love_you** (ASL ILY handshape, 2 CC BY 4.0 Roboflow sets) | ~10–20 min |

**`roboflow_ily`** needs a free Roboflow API key in `ROBOFLOW_API_KEY`
(roboflow.com → account → Settings → API → Private API Key) and
`pip install roboflow`. It downloads each project's COCO export to `data/
_ext_tmp/`, crops to the ILY-hand bbox (detection sets) or uses the whole image
(classification sets), extracts, and deletes the export. After this lands,
remove `i_love_you` from `PENDING_DATA` / `AUG_ONLY` in `build_dataset.py` and
rebuild — it then flows through the normal external path.

**`heart` via `coco_pose`**: `_classify_coco` now has a heart branch (both
wrists above the eyes, hands near the midline, both elbows outboard). Yield is
unverified — could be low. Spot-check `data/samples_ext/heart/` after and, if
too thin, keep `heart` in `PENDING_DATA`.

**`hagrid_v2_heart`** streams the `testdummyvt/hagRIDv2_512px` **val** split
(184 parquet shards, ~50–100 MB each, deleted after each) and stops once
`mini_heart` hits 1800 — it does **not** download the 42 GB dataset. Expect it
to chew through ~40–70 shards before it fills.

**`coco_pose`** downloads only `person_keypoints_train2017.json` (~250 MB),
filters it for arms-out / arm-raised / arms-down poses, then downloads **only**
the few hundred matching images one at a time (deleted after each) and runs
MediaPipe on them.

Still needs the user:
- `squat` → a Kaggle gym-workout video dataset (`~/.kaggle/kaggle.json`). Not
  wired yet; falls back to aug(s01).
- `sit` / `laying` → no clean streamable source found (NTU needs registration,
  Le2i's lab link is dead). `sit` falls back to aug(s01 platform-sits) with
  s02 `sit_03` as the cross-person test; `laying` is s02-only → aug-only like
  the other 3 stuck classes.

## Subject ids

The HF HaGRID samples carry only image + label, no worker id. `extract_external.py`
tags each image `<source>_<running index>` — since HaGRID has ~37 k subjects over
552 k images, a random image is ~a distinct person, so a random split of these
rows is effectively a subject split. Good enough; noted in the dataset card.
