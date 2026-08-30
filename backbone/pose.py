"""Thin wrapper around the body-pose backbone.

Locked at Phase 1: **MediaPipe Pose via the Tasks API** (`pose_landmarker`,
lite model, VIDEO running mode). 33-landmark BlazePose topology, CPU, 0 VRAM.
See docs/phase1_report.md for the benchmark. The `yolo` backend is kept only
for the Phase 1 comparison in scripts/phase1_eval.py.

Interface (stable — the rest of the pipeline depends on it):
    est = PoseEstimator()                  # backend="mediapipe" by default
    est.new_sequence()                     # call before each independent clip
    kp  = est.estimate(frame_bgr)          # -> Keypoints or None if no person
    kp.xy            (33, 2) float pixel coords
    kp.visibility    (33,)  float 0..1
    kp.left_wrist / kp.right_wrist / kp.shoulder_width
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
# subset of POSE_CONNECTIONS, enough to draw a readable skeleton
_MP_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 2), (0, 5), (9, 10),
]

# --- YOLO-pose (COCO, 17 keypoints) — Phase 1 comparison only ---
_YOLO_LEFT_WRIST, _YOLO_RIGHT_WRIST = 9, 10
_YOLO_LEFT_SHOULDER, _YOLO_RIGHT_SHOULDER = 5, 6
_COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

_FRAME_DT_MS = 33  # nominal; VIDEO mode only needs monotonic-increasing timestamps


class PoseEstimator:
    def __init__(self, backend: str = "mediapipe", min_confidence: float = 0.5):
        self.backend = backend
        if backend == "mediapipe":
            from mediapipe.tasks.python import BaseOptions, vision

            from .assets import asset_bytes

            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=asset_bytes("pose_landmarker_lite")),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_confidence,
                min_pose_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            self._model = vision.PoseLandmarker.create_from_options(opts)
            self._ts = 0
            self.edges = _MP_EDGES
            self._lw, self._rw = _MP_LEFT_WRIST, _MP_RIGHT_WRIST
            self._ls, self._rs = _MP_LEFT_SHOULDER, _MP_RIGHT_SHOULDER
        elif backend == "yolo":
            from ultralytics import YOLO

            self._model = YOLO("yolo11n-pose.pt")
            self._conf = min_confidence
            self.edges = _COCO_EDGES
            self._lw, self._rw = _YOLO_LEFT_WRIST, _YOLO_RIGHT_WRIST
            self._ls, self._rs = _YOLO_LEFT_SHOULDER, _YOLO_RIGHT_SHOULDER
        else:
            raise ValueError(f"unknown pose backend: {backend!r}")

    def new_sequence(self) -> None:
        """Reset temporal state. Call between independent clips so tracking from
        one clip does not leak into the next (VIDEO mode requires monotonic
        timestamps, so we also bump past the last one)."""
        if self.backend == "mediapipe":
            self._ts += 10 * _FRAME_DT_MS

    def estimate(self, frame_bgr: np.ndarray) -> Keypoints | None:
        h, w = frame_bgr.shape[:2]
        if self.backend == "mediapipe":
            import cv2
            import mediapipe as mp

            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            self._ts += _FRAME_DT_MS
            res = self._model.detect_for_video(img, self._ts)
            if not res.pose_landmarks:
                return None
            lm = res.pose_landmarks[0]
            xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            vis = np.array([p.visibility for p in lm], dtype=np.float32)
        else:  # yolo
            res = self._model(frame_bgr, conf=self._conf, verbose=False)[0]
            kpts = res.keypoints
            if kpts is None or kpts.xy is None or kpts.xy.shape[0] == 0:
                return None
            xy = kpts.xy[0].cpu().numpy().astype(np.float32)
            if xy.shape[0] == 0:
                return None
            vis = (
                kpts.conf[0].cpu().numpy().astype(np.float32)
                if kpts.conf is not None
                else np.ones(len(xy), dtype=np.float32)
            )
        return Keypoints(xy, vis, self._lw, self._rw, self._ls, self._rs)
