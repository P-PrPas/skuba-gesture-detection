"""Phase 2 step 1: labelled clips -> per-frame normalized feature vectors.

    python -m pipeline.extract_features                # all clips in segments.csv
    python -m pipeline.extract_features --clip ok_01   # just one

For each clip: decode frames, run MediaPipe Pose + wrist-anchored Hands,
normalize (body + each hand), fuse -> 152-vector. Writes one .npz per clip to
data/features/ with X + subject/session/clip/class/frame metadata.

Frames where no body is detected are dropped (can't normalize) and counted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from backbone.hands import HandLandmarker, wrist_crop_box
from backbone.pose import PoseEstimator
from features.fuse import fuse
from features.normalize import normalize_body, normalize_hand
from features.schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "data" / "clips"
OUT_DIR = ROOT / "data" / "features"
SEGMENTS = ROOT / "data" / "segments.csv"

POSE_BACKBONE = "mediapipe"  # locked for the deployment target (see features/schema.py)


def _clip_path(clip_id: str, cls: str) -> Path:
    sub = cls if cls in CLASSES else "_review"
    return CLIPS_DIR / sub / f"{clip_id}.mp4"


def extract_clip(row: dict, pose: PoseEstimator, hands: HandLandmarker) -> dict | None:
    path = _clip_path(row["clip_id"], row["class"])
    if not path.exists():
        print(f"  MISSING {path} - run scripts/cut_segments.py")
        return None
    cap = cv2.VideoCapture(str(path))
    if hasattr(pose, "new_sequence"):
        pose.new_sequence()  # don't let tracking leak between clips
    vecs, frames = [], []
    fi, no_body = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        kp = pose.estimate(frame)
        if kp is None:
            no_body += 1
            continue
        body_norm = normalize_body(kp.xy)
        scale = kp.shoulder_width or 80.0
        hand_norms = []
        for wrist in (kp.left_wrist, kp.right_wrist):
            x0, y0, x1, y1 = wrist_crop_box(wrist, scale, frame.shape)
            crop = frame[y0:y1, x0:x1]
            lm = hands.detect(crop)
            hand_norms.append(normalize_hand(lm) if lm is not None else None)
        vecs.append(fuse(body_norm, hand_norms[0], hand_norms[1]))
        frames.append(fi)
    cap.release()
    if not vecs:
        print(f"  {row['clip_id']}: 0 usable frames ({no_body} no-body)")
        return None
    print(f"  {row['clip_id']}: {len(vecs)} frames ({no_body} dropped, no body)")
    n = len(vecs)
    return {
        "X": np.stack(vecs).astype(np.float32),
        "frame_idx": np.array(frames, dtype=np.int32),
        "clip_id": np.array([row["clip_id"]] * n),
        "subject_id": np.array([row["subject_id"]] * n),
        "session_id": np.array([row["session_id"]] * n),
        "label": np.array([row["class"]] * n),
        "no_body_frames": no_body,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="only this clip_id")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGMENTS, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["class"] in CLASSES]
    if args.clip:
        rows = [r for r in rows if r["clip_id"] == args.clip]

    pose = PoseEstimator(POSE_BACKBONE)
    hands = HandLandmarker()

    summary = []
    for row in rows:
        rec = extract_clip(row, pose, hands)
        if rec is None:
            continue
        out = OUT_DIR / f"{row['clip_id']}.npz"
        np.savez_compressed(out, **{k: v for k, v in rec.items() if k != "no_body_frames"})
        summary.append(
            {
                "clip_id": row["clip_id"],
                "label": row["class"],
                "frames": int(rec["X"].shape[0]),
                "no_body_frames": rec["no_body_frames"],
            }
        )

    (OUT_DIR / "_extract_summary.json").write_text(
        json.dumps({"pose_backbone": POSE_BACKBONE, "clips": summary}, indent=2)
    )
    total = sum(s["frames"] for s in summary)
    print(f"\n{len(summary)} clips, {total} frames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
