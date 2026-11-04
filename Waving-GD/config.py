"""Shared paths and constants for Waving-GD (raise-hand detector)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
RAW = ROOT / "data" / "raw"
KPTS = ROOT / "data" / "kpts"          # per-clip extracted keypoints (npz)
DATASET = ROOT / "data" / "dataset"    # built train/val tables
RUNS = ROOT / "outputs" / "runs"         # one folder per trained version: model.pt, model.json, train.log, eval_test.txt, videos/


def model_path(name: str) -> Path:
    return RUNS / name / "model.pt"
TEST_CLIPS = ROOT / "dataset"          # user's held-out clips (never trained on)

POSE_MODEL = "yolo11s-pose"
POSE_IMGSZ = 640
POSE_CONF = 0.25
KPT_VIS = 0.30      # keypoints below this confidence are zeroed in features
SMOOTH_WINDOW = 5   # per-track temporal smoothing (frames)
