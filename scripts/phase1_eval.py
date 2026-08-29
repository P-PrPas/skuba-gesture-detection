"""Phase 1 backbone sanity check (IMPLEMENTATION_PLAN.md Phase 1).

Runs the candidate body-pose backbones on the project's hard cases and
MediaPipe Hands on the overlapping-finger hand shapes, saves annotated
montages for visual review, and measures latency on THIS machine (used as a
proxy for the Ubuntu deploy laptop - see the note in the report).

    python scripts/phase1_eval.py --out scratch/phase1

Hard cases (from data/clips/, subject at ~2-4 m in the footage):
  posture : laying_01, squat_01, sit_02
  hands   : rock_01, ok_01, i_love_you_01, two_finger_01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIPS = ROOT / "data" / "clips"

POSTURE = {
    "laying": CLIPS / "laying" / "laying_01.mp4",
    "squat": CLIPS / "squat" / "squat_01.mp4",
    "sit": CLIPS / "sit" / "sit_02.mp4",
}
HANDS = {
    "rock": CLIPS / "rock" / "rock_01.mp4",
    "ok": CLIPS / "ok" / "ok_01.mp4",
    "i_love_you": CLIPS / "i_love_you" / "i_love_you_01.mp4",
    "two_finger": CLIPS / "two_finger" / "two_finger_01.mp4",
}

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def sample_frames(path: Path, n: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1), n).astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def draw_skeleton(frame, xy, edges, conf=None, thr=0.3):
    f = frame.copy()
    for i, j in edges:
        if conf is not None and (conf[i] < thr or conf[j] < thr):
            continue
        p, q = xy[i].astype(int), xy[j].astype(int)
        cv2.line(f, tuple(p), tuple(q), (0, 255, 0), 3)
    for k, (x, y) in enumerate(xy):
        if conf is not None and conf[k] < thr:
            continue
        cv2.circle(f, (int(x), int(y)), 4, (0, 165, 255), -1)
    return f


# ---------------- pose backends ----------------
class MPPose:
    name = "mediapipe"

    def __init__(self):
        from backbone.pose import PoseEstimator

        self.est = PoseEstimator("mediapipe")
        self.edges = self.est.edges

    def infer(self, frame):
        kp = self.est.estimate(frame)
        if kp is None:
            return None, None
        return kp.xy, kp.visibility


class YoloPose:
    name = "yolo11n-pose"

    def __init__(self):
        from ultralytics import YOLO

        self.m = YOLO("yolo11n-pose.pt")
        self.edges = COCO_EDGES

    def infer(self, frame):
        r = self.m(frame, verbose=False, conf=0.3)[0]
        if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
            return None, None
        xy = r.keypoints.xy[0].cpu().numpy()
        cf = r.keypoints.conf
        cf = cf[0].cpu().numpy() if cf is not None else np.ones(len(xy))
        if xy.shape[0] == 0:
            return None, None
        return xy, cf


class RtmPose:
    name = "rtmpose-t (rtmlib lite)"

    def __init__(self):
        from rtmlib import Body

        self.m = Body(mode="lightweight", backend="onnxruntime", device="cpu")
        self.edges = COCO_EDGES

    def infer(self, frame):
        kpts, scores = self.m(frame)
        if kpts is None or len(kpts) == 0:
            return None, None
        return np.asarray(kpts[0]), np.asarray(scores[0])


def eval_posture(backends, out: Path, warm=3):
    rows = []
    for be in backends:
        for cls, path in POSTURE.items():
            if not path.exists():
                print(f"  missing {path}")
                continue
            frames = sample_frames(path, 8)
            for _ in range(warm):
                be.infer(frames[0])
            t0 = time.perf_counter()
            results = [be.infer(f) for f in frames]
            ms = (time.perf_counter() - t0) / len(frames) * 1000
            hit = sum(1 for xy, _ in results if xy is not None)
            tiles = [
                draw_skeleton(fr, xy, be.edges, cf) if xy is not None else fr
                for fr, (xy, cf) in zip(frames, results)
            ]
            _montage(tiles, out / f"posture_{cls}_{be.name.split()[0]}.jpg")
            rows.append((be.name, cls, f"{hit}/{len(frames)}", f"{ms:.0f} ms/frame"))
    return rows


def eval_hands(out: Path, warm=3):
    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    from smoke_test import HAND_EDGES

    pose = PoseEstimator("mediapipe")
    hands = HandLandmarker()
    rows = []
    for cls, path in HANDS.items():
        if not path.exists():
            continue
        frames = sample_frames(path, 8)
        crops, drawn, hit, times = [], [], 0, []
        for fr in frames:
            kp = pose.estimate(fr)
            if kp is None:
                continue
            wrist = kp.left_wrist if kp.left_wrist[1] < kp.right_wrist[1] else kp.right_wrist
            x0, y0, x1, y1 = wrist_crop_box(wrist, kp.shoulder_width or 90, fr.shape)
            crop = fr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            t0 = time.perf_counter()
            lm = hands.detect(crop)
            times.append((time.perf_counter() - t0) * 1000)
            c = cv2.resize(crop, (200, 200))
            if lm is not None:
                hit += 1
                s = np.array([200 / crop.shape[1], 200 / crop.shape[0]])
                lm2 = lm * s
                for a, b in HAND_EDGES:
                    cv2.line(c, tuple(lm2[a].astype(int)), tuple(lm2[b].astype(int)), (255, 0, 255), 1)
                for x, y in lm2:
                    cv2.circle(c, (int(x), int(y)), 2, (0, 255, 255), -1)
            drawn.append(c)
        _montage(drawn, out / f"hands_{cls}.jpg", cols=len(drawn) or 1)
        rows.append((cls, f"{hit}/{len(drawn)}", f"{np.mean(times):.0f} ms/crop" if times else "-"))
    return rows


def eval_combined_latency(clip=POSTURE["squat"], warm=5):
    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator

    pose = PoseEstimator("mediapipe")
    hands = HandLandmarker()
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    for f in frames[:warm]:
        pose.estimate(f)
    t0 = time.perf_counter()
    for f in frames:
        kp = pose.estimate(f)
        if kp is None:
            continue
        for wrist in (kp.left_wrist, kp.right_wrist):
            x0, y0, x1, y1 = wrist_crop_box(wrist, kp.shoulder_width or 90, f.shape)
            c = f[y0:y1, x0:x1]
            if c.size:
                hands.detect(c)
    dt = (time.perf_counter() - t0) / len(frames)
    return dt * 1000, 1.0 / dt, len(frames)


def _montage(tiles, path: Path, cols=8):
    if not tiles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w // 3), rows * (h // 3)), "black")
    for i, t in enumerate(tiles):
        im = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)).resize((w // 3, h // 3))
        sheet.paste(im, ((i % cols) * (w // 3), (i // cols) * (h // 3)))
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/phase1")
    args = ap.parse_args()
    out = Path(args.out)

    backends = []
    for cls in (MPPose, YoloPose, RtmPose):
        try:
            backends.append(cls())
        except Exception as e:  # noqa: BLE001
            print(f"skip {cls.__name__}: {e}")

    print("\n=== posture: keypoint plausibility + latency ===")
    for r in eval_posture(backends, out):
        print(f"  {r[0]:22s} {r[1]:8s} detect {r[2]:8s} {r[3]}")

    print("\n=== hands (MediaPipe Hands on wrist crops) ===")
    for r in eval_hands(out):
        print(f"  {r[0]:12s} detect {r[1]:8s} {r[2]}")

    ms, fps, n = eval_combined_latency()
    print(f"\n=== combined MediaPipe pose+2 hand crops (n={n}) ===")
    print(f"  {ms:.0f} ms/frame  ->  {fps:.1f} FPS  (this machine, CPU)")
    print(f"\nmontages -> {out}/")


if __name__ == "__main__":
    main()
