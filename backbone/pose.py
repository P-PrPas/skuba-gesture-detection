"""Thin wrapper around a body-pose backbone.

Phase 1 picks ONE backend and locks it (see IMPLEMENTATION_PLAN.md). Until then
both candidates live behind the same interface so the smoke test / hard-case
clips can be run against either with a flag.

Interface (stable — the rest of the pipeline depends on it):
    est = PoseEstimator(backend="mediapipe")
    kp  = est.estimate(frame_bgr)          # -> Keypoints or None if no person
    kp.xy            (N, 2) float pixel coords
    kp.visibility    (N,)  float 0..1
    kp.left_wrist / kp.right_wrist   -> (x, y) pixels
    est.edges        list[(i, j)] for drawing the skeleton
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoints:
    xy: np.ndarray          # (N, 2) pixels
    visibility: np.ndarray  # (N,)
    left_wrist_idx: int
    right_wrist_idx: int
    left_shoulder_idx: int
    right_shoulder_idx: int

    @property
    def left_wrist(self) -> np.ndarray:
        return self.xy[self.left_wrist_idx]

    @property
    def right_wrist(self) -> np.ndarray:
        return self.xy[self.right_wrist_idx]

    @property
    def shoulder_width(self) -> float:
        return float(
            np.linalg.norm(
                self.xy[self.left_shoulder_idx] - self.xy[self.right_shoulder_idx]
            )
        )


# --- MediaPipe Pose (BlazePose, 33 landmarks) ---
_MP_LEFT_WRIST, _MP_RIGHT_WRIST = 15, 16
_MP_LEFT_SHOULDER, _MP_RIGHT_SHOULDER = 11, 12

# --- YOLO-pose (COCO, 17 keypoints) ---
_YOLO_LEFT_WRIST, _YOLO_RIGHT_WRIST = 9, 10
_YOLO_LEFT_SHOULDER, _YOLO_RIGHT_SHOULDER = 5, 6
_COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


class PoseEstimator:
    def __init__(self, backend: str = "mediapipe", min_confidence: float = 0.5):
        self.backend = backend
        if backend == "mediapipe":
            import mediapipe as mp

            self._mp = mp
            self._model = mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            self.edges = list(mp.solutions.pose.POSE_CONNECTIONS)
            self._lw, self._rw = _MP_LEFT_WRIST, _MP_RIGHT_WRIST
            self._ls, self._rs = _MP_LEFT_SHOULDER, _MP_RIGHT_SHOULDER
        elif backend == "yolo":
            from ultralytics import YOLO

            self._model = YOLO("yolo11n-pose.pt")  # auto-downloads on first run
            self._conf = min_confidence
            self.edges = _COCO_EDGES
            self._lw, self._rw = _YOLO_LEFT_WRIST, _YOLO_RIGHT_WRIST
            self._ls, self._rs = _YOLO_LEFT_SHOULDER, _YOLO_RIGHT_SHOULDER
        else:
            raise ValueError(f"unknown pose backend: {backend!r}")

    def estimate(self, frame_bgr: np.ndarray) -> Keypoints | None:
        h, w = frame_bgr.shape[:2]
        if self.backend == "mediapipe":
            import cv2

            res = self._model.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            if lm is None:
                return None
            pts = lm.landmark
            xy = np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
            vis = np.array([p.visibility for p in pts], dtype=np.float32)
        else:  # yolo
            res = self._model(frame_bgr, conf=self._conf, verbose=False)[0]
            kpts = res.keypoints
            if kpts is None or kpts.xy is None or kpts.xy.shape[0] == 0:
                return None
            # first (most confident) person
            xy = kpts.xy[0].cpu().numpy().astype(np.float32)
            if xy.shape[0] == 0:  # box but no landmarks
                return None
            vis = (
                kpts.conf[0].cpu().numpy().astype(np.float32)
                if kpts.conf is not None
                else np.ones(len(xy), dtype=np.float32)
            )
        return Keypoints(xy, vis, self._lw, self._rw, self._ls, self._rs)
