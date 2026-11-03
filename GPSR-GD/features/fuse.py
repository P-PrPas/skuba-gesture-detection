"""Fuse normalized body + both hands into the single fixed-length feature vector
(ARCHITECTURE.md "Feature vector schema"). A missing hand -> zeros + presence 0.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    FEATURE_DIM,
    LH_OFF,
    LH_PRESENT,
    N_BODY,
    N_HAND,
    RH_OFF,
    RH_PRESENT,
)


def fuse(
    body_norm: np.ndarray,
    left_hand_norm: np.ndarray | None,
    right_hand_norm: np.ndarray | None,
) -> np.ndarray:
    """body_norm: (33,2); hands: (21,2) or None. -> (152,) float32."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[0 : N_BODY * 2] = body_norm.reshape(-1)
    if left_hand_norm is not None:
        v[LH_OFF : LH_OFF + N_HAND * 2] = left_hand_norm.reshape(-1)
        v[LH_PRESENT] = 1.0
    if right_hand_norm is not None:
        v[RH_OFF : RH_OFF + N_HAND * 2] = right_hand_norm.reshape(-1)
        v[RH_PRESENT] = 1.0
    return v
