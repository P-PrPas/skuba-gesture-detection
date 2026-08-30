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
- **Phase 1** 🟡 MediaPipe Pose (Tasks API, lite) + Hands chosen — CPU+GPU
  benchmark of 8 backbones (MediaPipe Solutions/Tasks×3, YOLO11n/s, RTMPose-t/m).
  MediaPipe Tasks-lite is 25 ms/frame on CPU — as fast as YOLO11n on CUDA, 0 VRAM,
  no keypoint bugs. GPU buys nothing for pose. Report:
  `results/phase1/backbone_report.docx` (+ `docs/phase1_report.md`,
  `scripts/phase1_eval.py`). Open: no clean `laying` clip yet; confirm FPS on the
  Acer; port `backbone/pose.py` from Solutions → Tasks API.
- **Phase 2** 🟡 extraction + augmentation pipeline done and run. Dataset built from
  `data/main.MOV` (1 subject, 1 session → **1323 frames, 16 clips, 14 classes**).
  Exit criterion **NOT met**: needs ≥3 subjects for a valid subject-wise
  train/val/test split (`dataset_card.json` → `phase2_exit_met: false`).
  Everything is in `train` until more subjects are recorded.

## Setup

This dev box's AV (AVG) does TLS interception, so pip needs the extra root.
Any Python 3.9–3.13 works (mediapipe 1.0.1 is a pure `py3-none` wheel).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
python scripts/fix_certs.py                 # append AV root to certifi (for .task model downloads)
```

The GPU backbone comparison (Phase 1 only) needs `requirements-phase1-gpu.txt`
on top; the core install above is all the pipeline itself uses.

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
