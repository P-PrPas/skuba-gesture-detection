"""Phase 0 smoke test — read a video/webcam stream and overlay body + hand keypoints.

    python smoke_test.py --source 0                # webcam
    python smoke_test.py --source data/main.MOV    # a clip
    python smoke_test.py --source data/main.MOV --pose yolo --save out.mp4

Hands are run on wrist-anchored crops taken from the body pose output (see
ARCHITECTURE.md), not on the full frame. A missing hand is just not drawn — the
classifier layer (later phases) is what handles presence flags.

Exit criterion for Phase 0: keypoints visibly track a person without crashing.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from backbone.hands import HandLandmarker, wrist_crop_box
from backbone.pose import PoseEstimator

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),
]


def draw_pose(frame, kp, edges):
    for i, j in edges:
        if kp.visibility[i] > 0.3 and kp.visibility[j] > 0.3:
            p, q = kp.xy[i].astype(int), kp.xy[j].astype(int)
            cv2.line(frame, tuple(p), tuple(q), (0, 255, 0), 2)
    for (x, y), v in zip(kp.xy, kp.visibility):
        if v > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)


def draw_hand(frame, landmarks, origin):
    ox, oy = origin
    for i, j in HAND_EDGES:
        p = (int(landmarks[i][0] + ox), int(landmarks[i][1] + oy))
        q = (int(landmarks[j][0] + ox), int(landmarks[j][1] + oy))
        cv2.line(frame, p, q, (255, 0, 255), 1)
    for x, y in landmarks:
        cv2.circle(frame, (int(x + ox), int(y + oy)), 2, (255, 255, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--pose", default="mediapipe", choices=["mediapipe", "yolo"])
    ap.add_argument("--save", default=None, help="write annotated video here instead of showing a window")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")

    pose = PoseEstimator(backend=args.pose)
    hands = HandLandmarker()

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if hasattr(pose, "new_sequence"):
        pose.new_sequence()
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        kp = pose.estimate(frame)
        if kp is not None:
            draw_pose(frame, kp, pose.edges)
            scale = kp.shoulder_width or 80.0
            for wrist in (kp.left_wrist, kp.right_wrist):
                x0, y0, x1, y1 = wrist_crop_box(wrist, scale, frame.shape)
                crop = frame[y0:y1, x0:x1]
                lm = hands.detect(crop)
                if lm is not None:
                    draw_hand(frame, lm, (x0, y0))
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

        fps_now = n / max(1e-6, time.time() - t0)
        cv2.putText(frame, f"{args.pose}  {fps_now:4.1f} fps", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("smoke_test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {args.save} ({n} frames)")
    cv2.destroyAllWindows()
    print(f"done: {n} frames, {n / max(1e-6, time.time() - t0):.1f} fps avg")


if __name__ == "__main__":
    main()
