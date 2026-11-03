"""Person-detector front-end (YOLO11n detection + ByteTrack IDs).

An alternative to MediaPipe's own multi-person detector for the live pipeline:
YOLO finds people much more reliably in cluttered / non-upright / partly
occluded scenes (main3.MOV: ~80 % of frames catch both people vs MediaPipe's
~29 %), at the cost of ~85-130 ms/frame on CPU. `DETECT_EVERY` amortises that —
run YOLO every K frames, hold the boxes between (fine for people who aren't
sprinting).

    det = PersonDetector()
    for frame in stream:
        for tid, (x0, y0, x1, y1) in det.detect(frame):
            ...

Weights (`yolo11n.pt`, ~5 MB) auto-download to the repo root on first use.
Enable via `config.PERSON_DETECTOR = "yolo"`.
"""

from __future__ import annotations

try:
    from config import DETECT_CONF, DETECT_EVERY, DETECT_IMGSZ, DETECT_MIN_AREA_FRAC
except Exception:
    DETECT_IMGSZ, DETECT_CONF, DETECT_EVERY, DETECT_MIN_AREA_FRAC = 512, 0.45, 2, 0.015


class PersonDetector:
    def __init__(self, imgsz=None, conf=None, every=None, min_area_frac=None):
        from ultralytics import YOLO

        self.model = YOLO("yolo11n.pt")               # detection, not -pose
        self.imgsz = imgsz or DETECT_IMGSZ
        self.conf = conf or DETECT_CONF
        self.every = max(1, every or DETECT_EVERY)
        self.min_area = (DETECT_MIN_AREA_FRAC if min_area_frac is None else min_area_frac)
        self._i = 0
        self._last: list[tuple[int, tuple]] = []

    def detect(self, frame_bgr):
        """-> [(track_id, (x0, y0, x1, y1)), ...]  (ints, frame pixels)."""
        if self._i % self.every == 0:
            r = self.model.track(frame_bgr, classes=[0], conf=self.conf,
                                 imgsz=self.imgsz, persist=True, verbose=False,
                                 tracker="bytetrack.yaml")[0]
            out = []
            b = r.boxes
            if b is not None and len(b):
                fh, fw = frame_bgr.shape[:2]
                min_a = self.min_area * fh * fw
                ids = (b.id.int().tolist() if b.id is not None
                       else list(range(len(b))))
                for tid, xyxy in zip(ids, b.xyxy.cpu().numpy()):
                    x0, y0, x1, y1 = (int(v) for v in xyxy)
                    if (x1 - x0) * (y1 - y0) >= min_a:        # drop tiny / spurious
                        out.append((int(tid), (x0, y0, x1, y1)))
            self._last = out
        self._i += 1
        return self._last
