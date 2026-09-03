"""Phase 5 real-time demo — webcam -> pose+hands -> classifier -> smoothing,
with a live OpenCV GUI (skeleton overlay, gesture label, confidence bar, the
smoothing window, and an FPS/latency readout).

    python -m pipeline.demo                       # webcam 0
    python -m pipeline.demo --video clip.mp4      # a video file, looped
    python -m pipeline.demo --benchmark           # headless latency/FPS report
    python -m pipeline.demo --window 7 --hand-every 2 --model mlp

Keys:  q quit   space pause   [ / ]  window N-/+   h  cycle hand-every (1/2/3)
       r reset smoother   s screenshot -> results/phase5/

`--hand-every K` runs the (expensive) hand landmarker every K frames and reuses
the last landmarks in between — the Phase 1 latency lever (hands are ~2/3 of a
frame). See docs/phase5.md.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from backbone.hands import HandLandmarker
from backbone.pose import PoseEstimator
from pipeline.runtime import GestureClassifier, TemporalSmoother, frame_to_vec

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "results" / "phase5"

GREEN, GRAY, CYAN, ORANGE, WHITE = (
    (90, 220, 90), (150, 150, 150), (230, 200, 60), (60, 160, 255), (240, 240, 240))

# BlazePose-33 skeleton edges (matches backbone.pose._MP_EDGES)
_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 2), (0, 5), (9, 10)]


def _panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, 1 - alpha,
                                            np.zeros_like(sub), alpha, 0)


def draw(frame, kp, boxes, out, fps, tm, n, hand_every):
    h, w = frame.shape[:2]
    if kp is not None:
        for i, j in _EDGES:
            pi, pj = kp.xy[i].astype(int), kp.xy[j].astype(int)
            cv2.line(frame, tuple(pi), tuple(pj), GREEN, 2, cv2.LINE_AA)
        for p in kp.xy.astype(int):
            cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for b, lbl in zip(boxes or [], ("L", "R")):
        x0, y0, x1, y1 = b
        cv2.rectangle(frame, (x0, y0), (x1, y1), CYAN, 1)
        cv2.putText(frame, lbl, (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, CYAN, 1, cv2.LINE_AA)

    # gesture panel (top-left) — width scales to the label, font shrinks to fit
    is_idle = out.label in ("idle", "unknown")
    col = GRAY if is_idle else GREEN
    label = out.label.upper()
    fs = 1.3
    while cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)[0][0] > 380 and fs > 0.6:
        fs -= 0.1
    pw = max(300, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)[0][0] + 40)
    _panel(frame, 12, 12, pw, 116)
    cv2.putText(frame, label, (24, 58), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 3, cv2.LINE_AA)
    cv2.putText(frame, f"raw: {out.raw_label}", (24, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    # confidence bar
    bx, by, bw = 24, 74, pw - 90
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 16), (70, 70, 70), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * out.confidence), by + 16), col, -1)
    cv2.putText(frame, f"{out.confidence:.2f}", (bx + bw + 8, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # smoothing window dots (bottom-left)
    for k, lab in enumerate(out.window):
        c = GREEN if lab == out.label else ORANGE
        cv2.circle(frame, (28 + k * 20, h - 26), 7, c, -1, cv2.LINE_AA)
    cv2.putText(frame, f"N={n}  hand-every={hand_every}", (28, h - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # perf (top-right)
    txt = f"{fps:4.1f} FPS  |  {tm.get('total', 0):.0f} ms  " \
          f"(pose {tm.get('pose', 0):.0f} / hands {tm.get('hands', 0):.0f})"
    cv2.putText(frame, txt, (w - 12 - 8 * len(txt), 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "q quit  space pause  [ ] N  h hands  r reset  s shot",
                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200),
                1, cv2.LINE_AA)


def _open(args):
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {src!r}")
    return cap


def benchmark(args):
    cap = _open(args)
    pose, hands = PoseEstimator("mediapipe"), HandLandmarker()
    clf = GestureClassifier(args.model)
    for he in (1, 2, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev = (None, None)
        stages = {"total": [], "pose": [], "hands": [], "classify": []}
        i = got = warm = 0
        while got < args.frames:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if warm < 10:                       # let XNNPACK / caches settle
                frame_to_vec(frame, pose, hands, detect_hands=(i % he == 0),
                             prev_hand_norms=prev)
                warm += 1
                i += 1
                continue
            tmg, t0 = {}, time.perf_counter()
            v, _, _, hn = frame_to_vec(frame, pose, hands, timings=tmg,
                                       detect_hands=(i % he == 0), prev_hand_norms=prev)
            if i % he == 0:
                prev = hn
            if v is not None:
                tc = time.perf_counter()
                clf.predict(v)
                stages["classify"].append((time.perf_counter() - tc) * 1000)
                stages["total"].append((time.perf_counter() - t0) * 1000)
                for k in ("pose", "hands"):
                    stages[k].append(tmg.get(k, 0.0))
                got += 1
            i += 1
        p = lambda a, q: float(np.percentile(a, q)) if a else 0.0  # noqa: E731
        tot = stages["total"]
        print(f"\nhand-every={he}  ({len(tot)} frames with a body)")
        for k in ("total", "pose", "hands", "classify"):
            a = stages[k]
            print(f"  {k:9s} p50 {p(a, 50):6.1f}  p90 {p(a, 90):6.1f}  "
                  f"p99 {p(a, 99):6.1f} ms")
        print(f"  -> {1000 / max(p(tot, 50), 1e-6):.1f} FPS (p50)")
    cap.release()


def live(args):
    cap = _open(args)
    pose, hands = PoseEstimator("mediapipe"), HandLandmarker()
    clf = GestureClassifier(args.model)
    sm = TemporalSmoother(n=args.window, floor=clf.floor)
    hand_every = max(1, args.hand_every)
    prev_hn = (None, None)
    ft = deque(maxlen=30)
    paused = False
    i = 0
    SHOTS.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.record:                       # headless: annotate --video -> mp4
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, (w, h))
    print("demo running" + (f" -> {args.record}" if writer else
                            " — focus the window for keys"))

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.video and not writer:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if args.mirror and not args.video:
                frame = cv2.flip(frame, 1)

            tmg, t0 = {}, time.perf_counter()
            did_hands = i % hand_every == 0
            v, kp, boxes, hn = frame_to_vec(
                frame, pose, hands, timings=tmg,
                detect_hands=did_hands, prev_hand_norms=prev_hn)
            if did_hands:
                prev_hn = hn
            if v is not None:
                lab, conf, _ = clf.predict(v)
            else:
                lab, conf, kp, boxes = "idle", 0.0, None, None
            out = sm.update(lab, conf)
            tmg["total"] = (time.perf_counter() - t0) * 1000
            ft.append(time.perf_counter())
            i += 1

        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 else 0.0
        draw(frame, kp, boxes, out, fps, tmg, sm.n, hand_every)

        if writer:
            writer.write(frame)
            continue
        cv2.imshow("SKUBA gesture demo", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("["):
            sm.set_window(sm.n - 1)
        elif k == ord("]"):
            sm.set_window(sm.n + 1)
        elif k == ord("h"):
            hand_every = hand_every % 3 + 1
        elif k == ord("r"):
            sm.reset()
        elif k == ord("s"):
            fn = SHOTS / f"demo_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(fn), frame)
            print("saved", fn)

    cap.release()
    if writer:
        writer.release()
        print("wrote", args.record)
    else:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--window", type=int, default=7)   # phase5_flicker.py recommendation
    ap.add_argument("--hand-every", type=int, default=1, dest="hand_every")
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", action="store_false", dest="mirror")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--frames", type=int, default=200, help="benchmark frame count")
    ap.add_argument("--record", type=str, default=None,
                    help="headless: annotate --video into this .mp4 (no window)")
    args = ap.parse_args()
    (benchmark if args.benchmark else live)(args)


if __name__ == "__main__":
    main()
