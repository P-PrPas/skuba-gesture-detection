"""Realtime raise-hand detector: YOLO-pose (+ByteTrack) -> features -> MLP -> per-track smoothing.

Library use (this is what the ROS2 node wraps):
    det = RaiseHandDetector("v2_all_fpr05")
    for frame in stream:
        for r in det.update(frame):      # one dict per tracked person
            r["track_id"], r["cls"] (0/1), r["conf"], r["fps"], r["box"]

CLI demo:
    python -m pipeline.detector --model v2_all_fpr05 --source dataset/IMG_8243.MOV --render
    python -m pipeline.detector --model mlp_all --source 0 --show        # webcam
"""
from __future__ import annotations
import argparse, sys, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from classifier.model import Classifier
from features.keypoints import features_from_raw
from features.temporal import TrackSmoother


class RaiseHandDetector:
    def __init__(self, model_name: str, pose_model: str = C.POSE_MODEL, window: int = 3,
                 device: str | int = 0, half: bool = True, thr: float | None = None, min_box_frac: float = 0.0):
        self.pose = YOLO(str(C.WEIGHTS / f"{pose_model}.pt"))
        self.cls = Classifier(C.model_path(model_name))
        self.thr = self.cls.thr if thr is None else thr
        self.smooth = TrackSmoother(window); self.device, self.half = device, half
        self.min_box_frac = min_box_frac
        self._t = deque(maxlen=30)

    def update(self, frame: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        r = self.pose.track(frame, persist=True, imgsz=C.POSE_IMGSZ, conf=C.POSE_CONF, device=self.device,
                            verbose=False)[0]
        out, alive = [], set()
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else list(range(len(boxes)))
            kxy, kc = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
            H = frame.shape[0]
            feats, keep = [], []
            for j, b in enumerate(boxes):
                if b[3] - b[1] < self.min_box_frac * H:
                    continue
                feats.append(features_from_raw(np.concatenate([kxy[j], kc[j][:, None]], 1), b)); keep.append(j)
            if feats:
                p = self.cls.predict(np.stack(feats))
                for j, pj in zip(keep, p):
                    s = self.smooth.update(ids[j], float(pj)); alive.add(ids[j])
                    out.append(dict(track_id=ids[j], cls=int(s >= self.thr), conf=s, box=boxes[j], kpt=np.concatenate([kxy[j], kc[j][:, None]], 1)))
        self.smooth.drop(alive)
        self._t.append(time.perf_counter() - t0)
        fps = len(self._t) / sum(self._t)
        for o in out:
            o["fps"] = fps
        return out


def draw(frame, res):
    for o in res:
        x1, y1, x2, y2 = (int(v) for v in o["box"]); col = (0, 0, 255) if o["cls"] else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
        cv2.putText(frame, f"#{o['track_id']} {'RAISE' if o['cls'] else '-'} {o['conf']:.2f}", (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    if res:
        cv2.putText(frame, f"{res[0]['fps']:.1f} FPS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--pose-model", default=C.POSE_MODEL); ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame (simulate low FPS)")
    ap.add_argument("--save", help="explicit output path"); ap.add_argument("--render", action="store_true", help="save to outputs/runs/<model>/videos/<clip>.mp4")
    ap.add_argument("--show", action="store_true"); ap.add_argument("--device", default="0")
    a = ap.parse_args()
    det = RaiseHandDetector(a.model, a.pose_model, a.window, device=a.device if a.device == "cpu" else int(a.device), half=False)
    if a.render:
        a.save = str(C.RUNS / a.model / "videos" / f"{Path(a.source).stem}.mp4"); Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source); wr = None; n = 0; lat = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % a.stride:
            continue
        t0 = time.perf_counter(); res = det.update(f); lat.append(time.perf_counter() - t0)
        vis = draw(f, res)
        if a.save:
            if wr is None:
                wr = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), (cap.get(cv2.CAP_PROP_FPS) or 30) / a.stride, (vis.shape[1], vis.shape[0]))
            wr.write(vis)
        if a.show:
            cv2.imshow("raise-hand", cv2.resize(vis, None, fx=0.5, fy=0.5)); 
            if cv2.waitKey(1) == 27:
                break
    if lat:
        lat = np.array(lat[5:] or lat) * 1000
        print(f"{len(lat)} frames  latency ms: mean {lat.mean():.1f}  p95 {np.percentile(lat, 95):.1f}  -> {1000 / lat.mean():.1f} FPS (this machine)")
