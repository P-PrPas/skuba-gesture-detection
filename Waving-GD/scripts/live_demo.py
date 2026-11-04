"""Simple live GUI to try the raise-hand detector on a laptop (webcam or a video file).

    python scripts/live_demo.py                          # webcam 0, model v3_all_rec95
    python scripts/live_demo.py --source 1               # another camera
    python scripts/live_demo.py --source clip.mp4 --model v3_all_rec95
    python scripts/live_demo.py --headless --source dataset/IMG_8243.MOV --save out.mp4   # no window (for servers)

Keys:  q / Esc quit   space pause   m mirror view   k skeleton on/off   b boxes on/off   [ ] smoothing window -/+   - = threshold -/+
Window shows per person: track id, RAISE / not, smoothed probability; top-left: FPS, device, model, threshold.
Needs: pip install ultralytics torch opencv-python numpy   (not opencv-python-headless) and `git lfs pull` for the weights.
CPU-only laptops are slow (a few FPS); use --imgsz 480 to speed up (accuracy may drop - the model was trained at 640).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from pipeline.detector import RaiseHandDetector, resolve_device
from features.temporal import TrackSmoother

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]
RED, GREEN, WHITE = (0, 0, 255), (0, 200, 0), (255, 255, 255)


def text(img, s, org, scale=0.8, col=WHITE, th=2):
    w = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0][0]
    org = (int(min(max(org[0], 5), max(img.shape[1] - w - 5, 5))), int(org[1]))   # keep inside the frame
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def draw(img, res, thr, skel, boxes):
    s = max(img.shape[0] / 720, 0.6)
    for o in res:
        col = RED if o["conf"] >= thr else GREEN
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        if skel:
            k = o["kpt"]
            for a, b in SKELETON:
                if k[a, 2] > C.KPT_VIS and k[b, 2] > C.KPT_VIS:
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), col, max(int(2 * s), 1))
            for j in range(17):
                if k[j, 2] > C.KPT_VIS:
                    cv2.circle(img, tuple(k[j, :2].astype(int)), int((7 if j in (9, 10) else 3) * s), (0, 255, 255) if j in (9, 10) else col, -1)
        if boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, max(int(3 * s), 2))
        label = f"#{o['track_id']} {'RAISE HAND' if o['conf'] >= thr else 'no'}  {o['conf']:.2f}"
        text(img, label, (x1, max(y1 - 10, int(30 * s))), 0.9 * s, col, 2)
        # probability bar under the box
        bw = max(x2 - x1, 40); cv2.rectangle(img, (x1, y2 + 6), (x1 + bw, y2 + 18), (60, 60, 60), -1)
        cv2.rectangle(img, (x1, y2 + 6), (x1 + int(bw * min(o["conf"], 1)), y2 + 18), col, -1)
        cv2.line(img, (x1 + int(bw * thr), y2 + 3), (x1 + int(bw * thr), y2 + 21), WHITE, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--model", default="v3_all_rec95", help="folder name under outputs/runs/")
    ap.add_argument("--pose-model", default=C.POSE_MODEL); ap.add_argument("--device", default="auto")
    ap.add_argument("--window", type=int, default=3, help="smoothing window (frames)")
    ap.add_argument("--imgsz", type=int, default=C.POSE_IMGSZ)
    ap.add_argument("--no-mirror", action="store_true", help="webcam is shown mirrored by default")
    ap.add_argument("--width", type=int, default=0, help="resize frames to this width before inference (0 = keep)")
    ap.add_argument("--headless", action="store_true", help="no window (use with --save)"); ap.add_argument("--save")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    dev = resolve_device(a.device)
    det = RaiseHandDetector(a.model, a.pose_model, a.window, device=dev, half=False, imgsz=a.imgsz)
    thr = det.thr
    is_cam = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        sys.exit(f"cannot open source {a.source!r}")
    mirror = is_cam and not a.no_mirror; skel = boxes = True; paused = False; writer = None; n = 0; frame = None
    win = "raise-hand test (q to quit)"
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            if a.width and raw.shape[1] > a.width:
                raw = cv2.resize(raw, (a.width, int(raw.shape[0] * a.width / raw.shape[1])))
            if mirror:
                raw = cv2.flip(raw, 1)
            res = det.update(raw)
            frame = raw
        vis = draw(frame.copy(), res, thr, skel, boxes)
        fps = res[0]["fps"] if res else 0.0
        text(vis, f"{fps:4.1f} FPS | {dev} | {a.model} | thr {thr:.2f} | win {det.smooth.window}" + (" | PAUSED" if paused else ""), (10, 28), 0.7)
        text(vis, "no person" if not res else f"{sum(o['conf'] >= thr for o in res)}/{len(res)} raising", (10, 58), 0.7)
        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 25, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
        if a.headless:
            continue
        cv2.imshow(win, vis)
        k = cv2.waitKey(1 if not paused else 30) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord(" "): paused = not paused
        elif k == ord("m"): mirror = not mirror
        elif k == ord("k"): skel = not skel
        elif k == ord("b"): boxes = not boxes
        elif k == ord("["): det.smooth = TrackSmoother(max(det.smooth.window - 1, 1))
        elif k == ord("]"): det.smooth = TrackSmoother(det.smooth.window + 1)
        elif k == ord("-"): thr = max(thr - 0.05, 0.05)
        elif k == ord("="): thr = min(thr + 0.05, 0.95)
    cap.release(); writer and writer.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
