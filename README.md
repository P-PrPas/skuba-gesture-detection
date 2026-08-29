# skuba-gesture-detection

Gesture and posture recognition for a service robot. Pretrained pose/hand
backbones extract keypoints; a single classifier on the fused, normalized
feature vector decides the class (or `idle`). See `docs/` for the full picture:

- `docs/ARCHITECTURE.md` — feature schema, normalization, augmentation, smoothing
- `docs/IMPLEMENTATION_PLAN.md` — phased roadmap
- `docs/DATA_COLLECTION_SPEC.md` — recording protocol
- `CLAUDE.md` — hard constraints (read before changing architecture)

## Status

**Phase 0 — environment & scaffolding.** Backbones installed, smoke test runs.

## Setup

The machine's AV (AVG) does TLS interception, so pip needs the extra root.
Python 3.11 (mediapipe has no 3.12+ wheels yet).

```bash
python -m venv .venv                       # use a 3.11 interpreter
.venv/Scripts/python -m pip install -r requirements.txt \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
python scripts/fix_certs.py                 # append AV root to certifi (for model downloads)
```

## Smoke test (Phase 0 deliverable)

```bash
python smoke_test.py --source 0                          # webcam
python smoke_test.py --source data/main.MOV --pose yolo  # a clip, YOLO-pose
python smoke_test.py --source data/main.MOV --save out.mp4 --max-frames 200
```

Overlays body skeleton + wrist-anchored hand landmarks. `--pose {mediapipe,yolo}`
selects the body backbone (Phase 1 picks one and locks it).

## Dataset

`data/main.MOV` is one long take covering every class. `data/segments.csv` is
the hand-checked label key (start/end seconds + confidence). Regenerate the
per-class sub-clips with:

```bash
python scripts/cut_segments.py data/main.MOV data/segments.csv data/clips
```

Clips with an empty `class` land in `data/clips/_review/` for you to label.

## Layout

```
backbone/   thin wrappers around pose/hand models — no training code
docs/       design docs
data/       raw clips + extracted features (git-ignored, versioned separately)
```
`features/`, `classifier/`, `pipeline/` land in later phases.
