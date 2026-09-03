"""Phase 5 runtime — the shared per-frame pipeline + temporal smoothing.

- `frame_to_vec()` — raw BGR frame → the 152-d fused vector, the *exact* same
  extraction `pipeline/extract_features.py` uses offline, so live features and
  training features cannot drift.
- `GestureClassifier` — wraps the locked classifier bundle → (label, conf, proba).
- `TemporalSmoother` — sliding-window majority vote + the idle/unknown fallback
  (`docs/ARCHITECTURE.md` "Temporal smoothing").

`demo()` at the bottom is the runnable self-check (no camera needed).
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backbone.hands import wrist_crop_box
from features.fuse import fuse
from features.normalize import normalize_body, normalize_hand
from features.schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "classifier" / "models"


def frame_to_vec(frame_bgr, pose, hands, *, detect_hands=True,
                 prev_hand_norms=(None, None), timings=None):
    """→ (vec152, kp, [lh_box, rh_box], (lh_norm, rh_norm), (lh_lm_xy, rh_lm_xy)).
    vec152/kp are None when no body is found. `*_lm_xy` are the raw 21-point hand
    landmarks in FRAME pixels (or None) — for drawing only. `detect_hands=False`
    reuses `prev_hand_norms`. `timings` dict, if given, gets "pose"/"hands" ms."""
    t0 = time.perf_counter()
    kp = pose.estimate(frame_bgr)
    if timings is not None:
        timings["pose"] = (time.perf_counter() - t0) * 1000.0
    if kp is None:
        return None, None, None, (None, None), (None, None)
    v, boxes, hn, lm = vec_from_kp(frame_bgr, kp, hands, detect_hands=detect_hands,
                                  prev_hand_norms=prev_hand_norms, timings=timings)
    return v, kp, boxes, hn, lm


def vec_from_kp(frame_bgr, kp, hands, *, detect_hands=True,
                prev_hand_norms=(None, None), timings=None):
    """One person's Keypoints → (vec152, [lh_box, rh_box], (lh_norm, rh_norm),
    (lh_lm_xy, rh_lm_xy)). `*_lm_xy` = raw 21-point hand landmarks in FRAME
    pixels (or None), for overlay only. `frame_to_vec` (single) and the
    multi-person path both call this."""
    body_norm = normalize_body(kp.xy)
    scale = kp.shoulder_width or 80.0
    boxes, hand_norms, hand_lms = [], [], []
    th = time.perf_counter()
    for i, wrist in enumerate((kp.left_wrist, kp.right_wrist)):
        box = wrist_crop_box(wrist, scale, frame_bgr.shape)
        boxes.append(box)
        if detect_hands:
            x0, y0, x1, y1 = box
            lm = hands.detect(frame_bgr[y0:y1, x0:x1])
            hand_norms.append(normalize_hand(lm) if lm is not None else None)
            hand_lms.append(lm + np.array([x0, y0], np.float32) if lm is not None else None)
        else:
            hand_norms.append(prev_hand_norms[i])
            hand_lms.append(None)
    if timings is not None:
        timings["hands"] = timings.get("hands", 0.0) + (time.perf_counter() - th) * 1000.0

    v = fuse(body_norm, hand_norms[0], hand_norms[1])
    return v, boxes, tuple(hand_norms), tuple(hand_lms)


class GestureClassifier:
    """The locked classifier bundle + the Phase 4 Track-E confidence floor."""

    def __init__(self, name: str = "mlp"):
        import joblib

        b = joblib.load(MODELS / f"{name}.joblib")
        self.clf = b["clf"]
        self.clip = b["clip"]
        self.feat_mode = b.get("features", "raw")
        self.classes = [CLASSES[int(c)] for c in b["clf"].classes_]
        tj = MODELS / f"{name}.thresholds.json"
        self.floor = (json.loads(tj.read_text()).get("global_floor", 0.0)
                      if tj.exists() else 0.0)

    def predict(self, vec152: np.ndarray):
        from features.derived import to_features

        x = np.clip(np.asarray(vec152, np.float32)[None], -self.clip, self.clip)
        p = self.clf.predict_proba(to_features(x, self.feat_mode))[0]
        i = int(p.argmax())
        return self.classes[i], float(p[i]), p


@dataclass
class SmoothOut:
    label: str            # what to act on (after majority vote + floor)
    confidence: float     # mean confidence of the winning class in the window
    raw_label: str        # this frame's un-smoothed prediction
    window: list          # recent raw labels, oldest→newest (for the UI)


class TemporalSmoother:
    """Last-N-frames majority vote. Emits the window's plurality class with its
    mean confidence; if that mean is below `floor`, emits `fallback` ("unknown")
    instead. A *confident* `idle` prediction still comes through as "idle" —
    "unknown" means "not sure", not "nothing happening". N and floor are the
    Phase 5 tuning knobs."""

    def __init__(self, n: int = 7, floor: float = 0.2, fallback: str = "unknown"):
        self.floor = floor
        self.fallback = fallback
        self._buf: deque = deque(maxlen=max(1, n))

    @property
    def n(self) -> int:
        return self._buf.maxlen

    def set_window(self, n: int) -> None:
        self._buf = deque(self._buf, maxlen=max(1, n))

    def reset(self) -> None:
        self._buf.clear()

    def update(self, label: str, conf: float) -> SmoothOut:
        self._buf.append((label, float(conf)))
        win_label = Counter(l for l, _ in self._buf).most_common(1)[0][0]
        cs = [c for l, c in self._buf if l == win_label]
        mean_c = sum(cs) / len(cs)
        out = win_label if mean_c >= self.floor else self.fallback
        return SmoothOut(out, mean_c, label, [l for l, _ in self._buf])


def demo() -> None:
    # smoother: plurality + floor
    s = TemporalSmoother(n=5, floor=0.5)
    for lab, c in [("ok", 0.9)] * 4 + [("idle", 0.9)]:
        r = s.update(lab, c)
    assert r.label == "ok" and r.raw_label == "idle", r          # 4/5 still ok
    for _ in range(5):
        r = s.update("idle", 0.9)
    assert r.label == "idle", r                                  # confident idle passes
    # low-confidence plurality -> "unknown", not "idle"
    s.reset()
    for _ in range(5):
        r = s.update("t_pose", 0.30)
    assert r.label == "unknown" and abs(r.confidence - 0.30) < 1e-6, r
    # window resize keeps recent history
    s.set_window(3)
    assert s.n == 3 and len(s._buf) == 3

    # classifier bundle loads and predicts on a zero vector
    clf = GestureClassifier("mlp")
    lab, conf, p = clf.predict(np.zeros(152, np.float32))
    assert lab in CLASSES and 0.0 <= conf <= 1.0 and len(p) == len(clf.classes)
    print(f"runtime.demo OK  (floor={clf.floor}, zero-vec -> {lab} {conf:.2f})")


if __name__ == "__main__":
    demo()
