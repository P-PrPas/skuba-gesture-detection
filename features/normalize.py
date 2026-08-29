"""Normalization (ARCHITECTURE.md "Normalization"). Applied to body and to each
hand separately, before fusion. Makes features invariant to camera distance and
subject/hand size.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_LEFT_HIP,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_HIP,
    POSE_RIGHT_SHOULDER,
)

_EPS = 1e-6


def normalize_body(xy: np.ndarray) -> np.ndarray:
    """xy: (33, 2) pixel coords -> (33, 2) centered on the shoulder-hip midpoint,
    scaled by the shoulder-mid <-> hip-mid distance."""
    shoulder_mid = (xy[POSE_LEFT_SHOULDER] + xy[POSE_RIGHT_SHOULDER]) / 2
    hip_mid = (xy[POSE_LEFT_HIP] + xy[POSE_RIGHT_HIP]) / 2
    center = (shoulder_mid + hip_mid) / 2
    scale = np.linalg.norm(shoulder_mid - hip_mid)
    return (xy - center) / max(scale, _EPS)


def normalize_hand(xy: np.ndarray) -> np.ndarray:
    """xy: (21, 2) crop-local pixel coords -> (21, 2) centered on the wrist,
    scaled by the wrist <-> middle-finger-MCP distance (palm width)."""
    center = xy[HAND_WRIST]
    scale = np.linalg.norm(xy[HAND_MIDDLE_MCP] - xy[HAND_WRIST])
    return (xy - center) / max(scale, _EPS)
