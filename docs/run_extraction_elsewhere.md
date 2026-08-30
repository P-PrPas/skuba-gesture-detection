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
!pip -q install "mediapipe==1.0.1" "opencv-contrib-python==4.11.0.86" "numpy>=1.26,<3" "datasets>=3" pyarrow
```

```python
# 2. run each source (resumable; re-run if a shard times out)
!python -m pipeline.extract_external --list
!python -m pipeline.extract_external hagrid
!python -m pipeline.extract_external hagrid_nogesture
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

| batch | sources | classes | ~Colab time |
|---|---|---|---|
| 1 (ready) | `hagrid`, `hagrid_nogesture` | ok, two_finger, rock, thumb, idle (+ ILY negative) | ~30–60 min CPU |
| 2 (todo) | COCO filter, Le2i fall | t_pose, raise_right_hand, sit, laying | needs the COCO/Le2i `SOURCES` entries written first |

`squat` needs a Kaggle gym-workout dataset (`~/.kaggle/kaggle.json`); `mini_heart`
needs HaGRID **v2** `hand_heart` (42 GB zip — Colab Pro or a scratch VM). Both
fall back to aug(s01) if not pulled.

## Subject ids

The HF HaGRID samples carry only image + label, no worker id. `extract_external.py`
tags each image `<source>_<running index>` — since HaGRID has ~37 k subjects over
552 k images, a random image is ~a distinct person, so a random split of these
rows is effectively a subject split. Good enough; noted in the dataset card.
