"""Thin wrapper around MediaPipe Hands (default hand-landmark backbone).

Run on a wrist-anchored crop, NOT the full frame (see ARCHITECTURE.md
"Hand cropping strategy"). One detector instance per hand side is fine; here we
keep a single instance and call it per crop.

    hl = HandLandmarker()
    lm = hl.detect(crop_bgr)   # -> (21, 2) landmarks in crop-pixel coords, or None
"""

from __future__ import annotations

import numpy as np

# MediaPipe Hands landmark indices (see ARCHITECTURE.md normalization)
WRIST = 0
MIDDLE_FINGER_MCP = 9
NUM_LANDMARKS = 21


class HandLandmarker:
    def __init__(self, min_confidence: float = 0.5):
        import mediapipe as mp

        self._model = mp.solutions.hands.Hands(
            static_image_mode=True,  # crops are independent, no temporal tracking
            max_num_hands=1,
            min_detection_confidence=min_confidence,
        )

    def detect(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        import cv2

        if crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        res = self._model.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        pts = res.multi_hand_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)


def wrist_crop_box(wrist_xy: np.ndarray, body_scale_px: float, frame_shape) -> tuple:
    """Square crop around the wrist, sized relative to a body-scale reference so it
    scales with the person's distance from the camera (ARCHITECTURE.md).

    Returns (x0, y0, x1, y1) clamped to the frame.
    """
    h, w = frame_shape[:2]
    half = max(20.0, 0.9 * body_scale_px)  # ~1.8x shoulder width total
    cx, cy = float(wrist_xy[0]), float(wrist_xy[1])
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(w, cx + half))
    y1 = int(min(h, cy + half))
    return x0, y0, x1, y1
