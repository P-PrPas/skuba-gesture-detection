"""Phase 5 real-time demo — webcam -> pose+hands -> classifier -> smoothing,
with a live OpenCV GUI (skeleton overlay, gesture label, confidence bar, the
smoothing window, and an FPS/latency readout).

    python -m pipeline.demo --poses 1             # webcam, single person (fastest)
    python -m pipeline.demo --poses 3             # webcam, up to 3 people (MediaPipe)
    python -m pipeline.demo --detector yolo       # webcam, YOLO person-detect front-end
    python -m pipeline.demo --video clip.mp4      # a video file, looped
    python -m pipeline.demo --benchmark           # headless latency/FPS report
    python -m pipeline.demo --video x.MOV --record out.mp4 --labels data/segments.csv
    python -m pipeline.demo --camera 1 --window 7 --hand-every 2 --floor 0.45

`--floor T`: below window-confidence T the smoother emits "unknown" (orange in
the GUI) — distinct from a *confident* "idle" (gray). `--labels segments.csv`
overlays the ground-truth class for a `--video` and prints a frame-accuracy
tally. `.MOV` rotation metadata is auto-applied; `--rotate {90,180,270}` if not.

Keys:  q quit   space pause   [ / ]  window N-/+   h  cycle hand-every (1/2/3)
       r reset smoother   s screenshot -> results/phase5/

`--hand-every K` runs the (expensive) hand landmarker every K frames and reuses
the last landmarks in between — the Phase 1 latency lever (hands are ~2/3 of a
frame). See docs/phase5.md.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from backbone.hands import HandLandmarker
from backbone.pose import PoseEstimator
from features.schema import CLASSES
from pipeline.multiperson import MultiPerson
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


def _load_truth(csv_path, video_name):
    """segments.csv -> [(start_s, end_s, class), ...] for one source video."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if Path(r["source_video"]).name == Path(video_name).name:
                rows.append((float(r["start_s"]), float(r["end_s"]), r["class"]))
    return sorted(rows)


def _truth_at(rows, t):
    for a, b, c in rows:
        if a <= t < b:
            return c
    return None


def draw(frame, kp, boxes, out, fps, tm, n, hand_every, gt=None):
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
    col = {"unknown": ORANGE, "idle": GRAY}.get(out.label, GREEN)
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

    # ground truth (top-centre) when --labels is given
    if gt is not None:
        gt_lbl, ok = gt
        gc = GREEN if ok else (60, 60, 235)
        t = f"GT: {gt_lbl or '-'}"
        tw = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        _panel(frame, w // 2 - tw // 2 - 14, 12, tw + 28, 44)
        cv2.putText(frame, t, (w // 2 - tw // 2, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, gc, 2, cv2.LINE_AA)

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


_ID_COLORS = [(90, 220, 90), (230, 170, 60), (200, 120, 230), (80, 200, 230),
              (120, 120, 240), (240, 220, 90)]


def _label_col(label):
    return {"unknown": ORANGE, "idle": GRAY}.get(label, GREEN)


def draw_people(frame, results, fps, tm, n, hand_every):
    h, w = frame.shape[:2]
    for r in results:
        sk = _ID_COLORS[r.id % len(_ID_COLORS)]
        for i, j in _EDGES:
            cv2.line(frame, tuple(r.kp.xy[i].astype(int)), tuple(r.kp.xy[j].astype(int)),
                     sk, 2, cv2.LINE_AA)
        x0, y0, x1, y1 = (int(v) for v in r.bbox)
        cv2.rectangle(frame, (x0, y0), (x1, y1), sk, 1)
        for b in r.boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), CYAN, 1)
        # label chip anchored just above the person's box (below if near the top)
        txt = f"P{r.id}  {r.out.label.upper()}  {r.out.confidence:.2f}"
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cy = y0 - 10 if y0 > 40 else y1 + 28
        cx = min(max(x0, 4), w - tw - 12)
        _panel(frame, cx - 4, cy - 22, tw + 12, 30)
        cv2.putText(frame, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _label_col(r.out.label), 2, cv2.LINE_AA)

    det = f"det {tm['detect']:.0f} / " if "detect" in tm else ""
    txt = f"{fps:4.1f} FPS | {tm.get('total', 0):.0f} ms " \
          f"({det}pose {tm.get('pose', 0):.0f} / hands {tm.get('hands', 0):.0f}) | {len(results)} ppl"
    cv2.putText(frame, txt, (w - 12 - 8 * len(txt), 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"num_poses={n}  hand-every={hand_every}   q quit  space pause",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def _open(args):
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {src!r}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # bake in .MOV rotation metadata
    return cap


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rot(frame, deg):
    return cv2.rotate(frame, _ROT[deg % 360]) if deg % 360 in _ROT else frame


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
            frame = _rot(frame, args.rotate)
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
    if args.detector:                               # CLI override of config
        import config as _cfg
        _cfg.PERSON_DETECTOR = None if args.detector == "mediapipe" else "yolo"
    try:
        from config import PERSON_DETECTOR
    except Exception:
        PERSON_DETECTOR = None
    pose = PoseEstimator("mediapipe", num_poses=args.poses)   # None -> config.NUM_POSES
    hands = HandLandmarker()
    multi = pose.num_poses > 1 or args.multi or PERSON_DETECTOR
    clf = GestureClassifier(args.model)
    sm = TemporalSmoother(n=args.window,
                          floor=args.floor if args.floor is not None else clf.floor)
    mp = MultiPerson(args.model, window=args.window, floor=args.floor) if multi else None
    hand_every = max(1, args.hand_every)
    prev_hn = (None, None)
    ft = deque(maxlen=30)
    paused = False
    i = 0
    SHOTS.mkdir(parents=True, exist_ok=True)

    truth = None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hits = tot_gt = 0
    if args.labels:
        truth = _load_truth(args.labels, args.video or "")
        print(f"ground truth: {len(truth)} labelled segments")

    writer = None
    fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    print("demo running" + (f" -> {args.record}" if args.record else
                            " — focus the window for keys"))

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.video and not args.record:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            frame = _rot(frame, args.rotate)
            if args.mirror and not args.video:
                frame = cv2.flip(frame, 1)
            if args.record and writer is None:   # size from the real (rotated) frame
                hh, ww = frame.shape[:2]
                for cc in ("avc1", "mp4v"):
                    writer = cv2.VideoWriter(args.record,
                                             cv2.VideoWriter_fourcc(*cc), fps_out, (ww, hh))
                    if writer.isOpened():
                        break

            tmg, t0 = {}, time.perf_counter()
            did_hands = i % hand_every == 0
            if multi:
                results = mp.process(frame, pose, hands, detect_hands=did_hands,
                                     timings=tmg)
            else:
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
        if multi:
            draw_people(frame, results, fps, tmg, pose.num_poses, hand_every)
        else:
            gt = None
            if truth is not None:
                g = _truth_at(truth, i / src_fps)
                gt = (g, g == out.label)
                if g in CLASSES:
                    tot_gt += 1
                    hits += int(g == out.label)
            draw(frame, kp, boxes, out, fps, tmg, sm.n, hand_every, gt=gt)

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
            (mp or sm).set_window((mp.window if mp else sm.n) - 1)
        elif k == ord("]"):
            (mp or sm).set_window((mp.window if mp else sm.n) + 1)
        elif k == ord("h"):
            hand_every = hand_every % 3 + 1
        elif k == ord("r"):
            (mp or sm).reset()
        elif k == ord("s"):
            fn = SHOTS / f"demo_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(fn), frame)
            print("saved", fn)

    cap.release()
    if tot_gt:
        print(f"vs ground truth: {hits}/{tot_gt} in-vocab labelled frames correct "
              f"({100 * hits / tot_gt:.1f}%)  (transitions + cut classes excluded)")
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
    ap.add_argument("--floor", type=float, default=None,
                    help="confidence floor below which -> 'unknown' "
                         "(default: the model's locked 0.20)")
    ap.add_argument("--hand-every", type=int, default=1, dest="hand_every")
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", action="store_false", dest="mirror")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames CW before processing (if AUTO metadata fails)")
    ap.add_argument("--poses", type=int, default=None,
                    help="max people to track this run (overrides config.NUM_POSES; "
                         "1 = single-person, fastest)")
    ap.add_argument("--multi", action="store_true",
                    help="force multi-person mode (auto-on when config.NUM_POSES > 1)")
    ap.add_argument("--detector", choices=["mediapipe", "yolo"], default=None,
                    help="person detector: yolo = YOLO11n front-end (overrides config)")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--frames", type=int, default=200, help="benchmark frame count")
    ap.add_argument("--record", type=str, default=None,
                    help="headless: annotate --video into this .mp4 (no window)")
    ap.add_argument("--labels", type=str, default=None,
                    help="segments.csv — overlay ground truth for --video + score it")
    args = ap.parse_args()
    (benchmark if args.benchmark else live)(args)


if __name__ == "__main__":
    main()
