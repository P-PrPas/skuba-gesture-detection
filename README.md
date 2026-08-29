# skuba-gesture-detection

Gesture and posture recognition for a service robot. Pretrained pose/hand
backbones extract keypoints; a single classifier on the fused, normalized
feature vector decides the class (or `idle`). See `docs/` for the full picture:

- `docs/ARCHITECTURE.md` — feature schema, normalization, augmentation, smoothing
- `docs/IMPLEMENTATION_PLAN.md` — phased roadmap
- `docs/DATA_COLLECTION_SPEC.md` — recording protocol
- `CLAUDE.md` — hard constraints (read before changing architecture)

## Status

- **Phase 0** ✅ environment, backbone wrappers, smoke test.
- **Phase 1** 🟡 MediaPipe Pose + Hands chosen — beat YOLO11n-pose & RTMPose on
  keypoint stability, VRAM (CPU-only), and latency on the hard cases
  (`docs/phase1_report.md`, `scripts/phase1_eval.py`). Open: no clean `laying`
  clip to test yet; latency (~10 FPS CPU combined) needs confirming on the Acer.
- **Phase 2** 🟡 extraction + augmentation pipeline done and run. Dataset built from
  `data/main.MOV` (1 subject, 1 session → **1323 frames, 16 clips, 14 classes**).
  Exit criterion **NOT met**: needs ≥3 subjects for a valid subject-wise
  train/val/test split (`dataset_card.json` → `phase2_exit_met: false`).
  Everything is in `train` until more subjects are recorded.

## Setup

The machine's AV (AVG) does TLS interception, so pip needs the extra root.
Python 3.11 (mediapipe has no 3.12+ wheels yet).

```bash
python -m venv .venv                       # use a 3.11 interpreter
.venv/Scripts/python -m pip install -r requirements.txt \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
python scripts/fix_certs.py                 # append AV root to certifi (for model downloads)
```

## Smoke test

```bash
python smoke_test.py --source 0                          # webcam
python smoke_test.py --source data/main.MOV --save out.mp4 --max-frames 200
```

Overlays body skeleton + wrist-anchored hand landmarks. `--pose {mediapipe,yolo}`
(mediapipe is the locked choice; yolo kept for comparison only).

## Dataset pipeline

`data/main.MOV` is one long take covering every class. `data/segments.csv` is
the hand-checked label key (`clip_id, subject_id, session_id, class, start_s,
end_s, ...`).

```bash
python scripts/cut_segments.py data/main.MOV data/segments.csv data/clips  # 1. CSV -> per-class clips
python -m pipeline.extract_features                                        # 2. clips -> data/features/<clip>.npz
python -m pipeline.build_dataset                                           # 3. -> data/dataset/{train,val,test}.npz + card
```

`data/dataset/dataset_card.json` records the feature schema, backbone, aug
params, per-class counts, split method, and whether the Phase 2 exit criterion
is met. To add more recordings: drop the video in `data/`, add rows to
`segments.csv` with a new `subject_id`, and re-run the three steps.

## Layout

```
backbone/    thin wrappers around pose/hand models — no training code
features/    schema, normalization, fusion, augmentation (the 152-d contract)
pipeline/    extract_features, build_dataset  (real-time loop lands in Phase 5)
scripts/     cut_segments, fix_certs
docs/        design docs
data/        raw clips, per-clip features, built dataset (git-ignored except segments.csv)
classifier/  Phase 3
```
