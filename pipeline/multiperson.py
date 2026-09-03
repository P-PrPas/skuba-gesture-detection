"""Phase 5 multi-person: pose.estimate_all() → per-person classify → a light
IoU tracker so every person keeps their own temporal-smoothing window.

    mp = MultiPerson(model="mlp")
    for frame in stream:
        for r in mp.process(frame, pose, hands):
            r.id, r.bbox, r.kp, r.boxes, r.out   # out = SmoothOut

`demo()` at the bottom is the runnable check (synthetic, no camera).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pipeline.runtime import GestureClassifier, TemporalSmoother, vec_from_kp

try:
    from config import (CONF_FLOOR, MAX_CLASSIFY, SMOOTH_WINDOW,
                        TRACK_IOU, TRACK_MAX_MISSED)
except Exception:                                    # sensible fallbacks
    SMOOTH_WINDOW, CONF_FLOOR, MAX_CLASSIFY = 7, 0.20, 0
    TRACK_IOU, TRACK_MAX_MISSED = 0.30, 12


def _person_detector_cfg():
    try:
        import config
        return getattr(config, "PERSON_DETECTOR", None)   # read fresh (CLI can override)
    except Exception:
        return None


def bbox_from_kp(kp, pad=0.08):
    """Axis-aligned bbox around the confidently-visible keypoints, padded."""
    xy, vis = kp.xy, kp.visibility
    m = vis > 0.3
    pts = xy[m] if m.sum() >= 4 else xy
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return float(x0 - px), float(y0 - py), float(x1 + px), float(y1 + py)


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Track:
    id: int
    bbox: tuple
    sm: TemporalSmoother
    prev_hn: tuple = (None, None)
    missed: int = 0
    out: object = None


@dataclass
class PersonResult:
    id: int
    bbox: tuple
    kp: object
    boxes: list
    out: object                                     # runtime.SmoothOut
    hand_lms: tuple = (None, None)                   # (21,2) frame-pixel, for overlay


class MultiPerson:
    def __init__(self, model: str = "mlp", window: int | None = None,
                 floor: float | None = None):
        self.clf = GestureClassifier(model)
        self.window = window or SMOOTH_WINDOW
        self.floor = self.clf.floor if floor is None else floor
        self._tracks: list[_Track] = []
        self._next_id = 0
        self.det = None
        if _person_detector_cfg() == "yolo":
            from backbone.detect import PersonDetector
            self.det = PersonDetector()
            self._sm: dict[int, TemporalSmoother] = {}   # yolo track id -> smoother
            self._prev_hn: dict[int, tuple] = {}

    def _new_track(self, bbox) -> _Track:
        t = _Track(self._next_id, bbox, TemporalSmoother(self.window, self.floor))
        self._next_id += 1
        self._tracks.append(t)
        return t

    def set_window(self, n: int) -> None:
        self.window = max(1, n)
        for t in self._tracks:
            t.sm.set_window(self.window)
        for sm in getattr(self, "_sm", {}).values():
            sm.set_window(self.window)

    def reset(self) -> None:
        self._tracks.clear()
        if self.det is not None:
            self._sm.clear(); self._prev_hn.clear()

    def process(self, frame_bgr, pose, hands, *, detect_hands=True, timings=None):
        if self.det is not None:
            return self._process_yolo(frame_bgr, pose, hands, detect_hands, timings)
        return self._process_mp(frame_bgr, pose, hands, detect_hands, timings)

    # --- YOLO-detect front-end: box+id per person, MediaPipe pose per crop ---
    def _process_yolo(self, frame_bgr, pose, hands, detect_hands, timings):
        import time as _t

        h, w = frame_bgr.shape[:2]
        td = _t.perf_counter()
        boxes = self.det.detect(frame_bgr)
        if timings is not None:
            timings["detect"] = (_t.perf_counter() - td) * 1000.0
        if MAX_CLASSIFY:
            boxes = sorted(boxes, key=lambda b: -(b[1][3] - b[1][1]))[:MAX_CLASSIFY]

        results, tp, th = [], 0.0, 0.0
        alive = set()
        for tid, (x0, y0, x1, y1) in boxes:
            bw, bh = x1 - x0, y1 - y0
            # generous margin that FOLLOWS the box shape: MediaPipe Pose degrades
            # or fails on a tight box (10% pad -> 14% no-pose, loses the
            # lying-person's orientation). 22% -> 10% no-pose and recovers
            # `laying`. A square crop is worse — it pulls in background/other
            # people for a wide (lying) box.
            pad = 0.22
            cx0 = max(0, int(x0 - pad * bw)); cy0 = max(0, int(y0 - pad * bh))
            cx1 = min(w, int(x1 + pad * bw)); cy1 = min(h, int(y1 + pad * bh))
            t0 = _t.perf_counter()
            kp = pose.estimate_crop(frame_bgr[cy0:cy1, cx0:cx1])
            tp += (_t.perf_counter() - t0) * 1000.0
            if kp is None:
                continue
            kp.xy[:, 0] += cx0; kp.xy[:, 1] += cy0          # crop -> frame coords
            # drop garbage poses (spurious tiny detections, MediaPipe wrap-around)
            sh_hip = float(np.linalg.norm(
                (kp.xy[11] + kp.xy[12]) / 2 - (kp.xy[23] + kp.xy[24]) / 2))
            if sh_hip < 0.06 * max(bw, bh) or (kp.visibility > 0.5).sum() < 12:
                continue
            t0 = _t.perf_counter()
            v, hboxes, hn, hlm = vec_from_kp(
                frame_bgr, kp, hands, detect_hands=detect_hands,
                prev_hand_norms=self._prev_hn.get(tid, (None, None)))
            th += (_t.perf_counter() - t0) * 1000.0
            if detect_hands:
                self._prev_hn[tid] = hn
            sm = self._sm.setdefault(tid, TemporalSmoother(self.window, self.floor))
            lab, conf, _ = self.clf.predict(v)
            out = sm.update(lab, conf)
            results.append(PersonResult(tid, (x0, y0, x1, y1), kp, hboxes, out, hlm))
            alive.add(tid)
        for tid in list(self._sm):                          # forget vanished tracks
            if tid not in alive:
                self._sm.pop(tid, None); self._prev_hn.pop(tid, None)
        if timings is not None:
            timings["pose"], timings["hands"] = tp, th
        return results

    def _process_mp(self, frame_bgr, pose, hands, detect_hands, timings):
        tp = time.perf_counter()
        people = pose.estimate_all(frame_bgr)        # largest-first, already ≤ num_poses
        if timings is not None:
            timings["pose"] = (time.perf_counter() - tp) * 1000.0
        if MAX_CLASSIFY:
            people = people[:MAX_CLASSIFY]
        dets = [(kp, bbox_from_kp(kp)) for kp in people]

        # greedy IoU association: best pairs first
        pairs = sorted(
            ((_iou(t.bbox, d[1]), ti, di) for ti, t in enumerate(self._tracks)
             for di, d in enumerate(dets)),
            reverse=True)
        used_t, used_d, matched = set(), set(), {}
        for iou, ti, di in pairs:
            if iou < TRACK_IOU or ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di); matched[di] = self._tracks[ti]

        results = []
        th = time.perf_counter()
        for di, (kp, bbox) in enumerate(dets):
            trk = matched.get(di) or self._new_track(bbox)
            trk.bbox, trk.missed = bbox, 0
            v, boxes, hn, hlm = vec_from_kp(frame_bgr, kp, hands, detect_hands=detect_hands,
                                           prev_hand_norms=trk.prev_hn)
            if detect_hands:
                trk.prev_hn = hn
            lab, conf, _ = self.clf.predict(v)
            trk.out = trk.sm.update(lab, conf)
            results.append(PersonResult(trk.id, bbox, kp, boxes, trk.out, hlm))
        if timings is not None:
            timings["hands"] = (time.perf_counter() - th) * 1000.0

        # age out tracks that got no detection this frame
        for ti, t in enumerate(self._tracks):
            if ti not in used_t and t not in matched.values():
                t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= TRACK_MAX_MISSED]
        return results


def demo() -> None:
    from types import SimpleNamespace

    from features.schema import FEATURE_DIM

    class _KP:
        def __init__(self, cx):
            self.xy = np.array([[cx, 100.0], [cx + 50, 300.0]] + [[cx, 200.0]] * 31,
                               dtype=np.float32)
            self.visibility = np.ones(33, np.float32)
            self.left_wrist = np.array([cx, 150.0]); self.right_wrist = np.array([cx + 20, 150.0])
            self.shoulder_width = 60.0

    a, b = _iou((0, 0, 10, 10), (0, 0, 10, 10)), _iou((0, 0, 10, 10), (20, 20, 30, 30))
    assert a == 1.0 and b == 0.0

    class _Pose:
        def __init__(self): self.n = 0
        def estimate_all(self, f):
            self.n += 1
            # two people, one drifts right over time; person B vanishes after frame 3
            ppl = [_KP(100 + self.n * 2)]
            if self.n <= 3:
                ppl.append(_KP(600))
            return ppl

    class _Hands:
        def detect(self, crop): return None

    mp = MultiPerson.__new__(MultiPerson)
    mp.clf = SimpleNamespace(floor=0.2,
                             predict=lambda v: ("ok" if v.sum() == 0 else "ok", 0.9, None))
    mp.window, mp.floor, mp._tracks, mp._next_id, mp.det = 5, 0.2, [], 0, None
    pose, hands = _Pose(), _Hands()
    ids_seen = set()
    for _ in range(8):
        rs = mp.process(np.zeros((720, 1280, 3), np.uint8), pose, hands)
        ids_seen |= {r.id for r in rs}
    # person A tracked as one id the whole time; B aged out (its id not re-used forever)
    assert len(ids_seen) == 2, ids_seen
    assert mp.process(np.zeros((720, 1280, 3), np.uint8), pose, hands)[0].out.label == "ok"
    print(f"multiperson.demo OK  (ids={sorted(ids_seen)}, tracks now {len(mp._tracks)})")


if __name__ == "__main__":
    demo()
