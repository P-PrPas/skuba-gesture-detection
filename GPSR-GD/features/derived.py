"""Derived features: joint angles + a few ratios, computed from the 152-d fused
vector. Angles are invariant to overall scale AND to the proportional skeleton
distortion MediaPipe Hands shows between a tight robot-camera crop and a large
webcam hand — that distortion is what broke `rock` in the Phase 3 baseline
(docs/phase3_baseline.md).

    from features.derived import to_features
    X2 = to_features(X_152, mode="both")   # "raw" | "derived" | "both"

`to_features` works on a single vector or a batch (N, 152).
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

_EPS = 1e-6

# ---- body (MediaPipe-33) triplets for joint angles: (a, vertex, c) ----
_BODY_ANGLES = [
    ("l_elbow", 11, 13, 15), ("r_elbow", 12, 14, 16),
    ("l_shoulder", 13, 11, 23), ("r_shoulder", 14, 12, 24),
    ("l_hip", 11, 23, 25), ("r_hip", 12, 24, 26),
    ("l_knee", 23, 25, 27), ("r_knee", 24, 26, 28),
]
# hand (MediaPipe-21) finger MCP triplets: (wrist/base, mcp, tip)
_FINGERS = [("thumb", 1, 2, 4), ("index", 0, 5, 8), ("middle", 0, 9, 12),
            ("ring", 0, 13, 16), ("pinky", 0, 17, 20)]
_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

N_BODY_DERIVED = len(_BODY_ANGLES) + 5     # 8 angles + torso lean + 2 wrist-heights + shoulder tilt + inter-wrist dist
N_HAND_DERIVED = 5 + 4 + 3                 # 5 curl + 4 spread + 3 ratios
DERIVED_DIM = N_BODY_DERIVED + 2 * N_HAND_DERIVED   # 12 + 24 = 36


def _angle(a, v, c):
    """Angle at vertex v (radians), batched. a,v,c: (N,2)."""
    u1 = a - v
    u2 = c - v
    n1 = np.linalg.norm(u1, axis=-1) + _EPS
    n2 = np.linalg.norm(u2, axis=-1) + _EPS
    cos = np.clip((u1 * u2).sum(-1) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos)


def _body_xy(X):
    return X[:, : N_BODY * 2].reshape(len(X), N_BODY, 2)


def _hand_xy(X, off):
    return X[:, off: off + N_HAND * 2].reshape(len(X), N_HAND, 2)


def _body_derived(X):
    b = _body_xy(X)
    out = [_angle(b[:, a], b[:, v], b[:, c]) for _, a, v, c in _BODY_ANGLES]
    sho_mid = (b[:, 11] + b[:, 12]) / 2
    hip_mid = (b[:, 23] + b[:, 24]) / 2
    spine = sho_mid - hip_mid
    out.append(np.arctan2(spine[:, 0], -spine[:, 1] + _EPS))          # torso lean vs vertical
    out.append(b[:, 15, 1] - sho_mid[:, 1])                           # L wrist height rel shoulders
    out.append(b[:, 16, 1] - sho_mid[:, 1])                           # R wrist height
    out.append(np.arctan2(b[:, 12, 1] - b[:, 11, 1], b[:, 12, 0] - b[:, 11, 0] + _EPS))  # shoulder tilt
    # inter-wrist distance (body units): each hand is normalised on its own wrist,
    # so the vector otherwise has no direct "hands together" signal — separates
    # mini_heart / heart (hands meet) from t_pose / raise_* (hands apart).
    out.append(np.linalg.norm(b[:, 15] - b[:, 16], axis=-1))
    return np.stack(out, axis=1)


def _hand_derived(X, off, present_idx):
    h = _hand_xy(X, off)
    present = X[:, present_idx] > 0.5
    feats = []
    # finger curl: angle at MCP between (base->mcp) and (mcp->tip)
    for _, base, mcp, tip in _FINGERS:
        feats.append(_angle(h[:, base], h[:, mcp], h[:, tip]))
    # spreads: angle at wrist between pairs of MCPs
    w = h[:, 0]
    feats.append(_angle(h[:, _MCP["thumb"]], w, h[:, _MCP["index"]]))   # thumb abduction
    feats.append(_angle(h[:, _MCP["index"]], w, h[:, _MCP["pinky"]]))   # full spread
    feats.append(_angle(h[:, _TIP["index"]], w, h[:, _TIP["pinky"]]))   # fingertip spread
    feats.append(_angle(h[:, _TIP["thumb"]], w, h[:, _TIP["pinky"]]))   # thumb-pinky (ILY)
    # ratios vs palm length (wrist -> middle MCP)
    palm = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1) + _EPS
    feats.append(np.linalg.norm(h[:, _TIP["thumb"]] - h[:, _TIP["index"]], axis=-1) / palm)
    feats.append(np.linalg.norm(h[:, _TIP["index"]] - h[:, _TIP["middle"]], axis=-1) / palm)
    feats.append(np.linalg.norm(h[:, _TIP["thumb"]] - h[:, _TIP["pinky"]], axis=-1) / palm)
    F = np.stack(feats, axis=1).astype(np.float32)
    F[~present] = 0.0                                                  # zero when hand absent
    return F


def derived(X: np.ndarray) -> np.ndarray:
    """(N,152) -> (N, DERIVED_DIM)."""
    return np.concatenate([
        _body_derived(X),
        _hand_derived(X, LH_OFF, LH_PRESENT),
        _hand_derived(X, RH_OFF, RH_PRESENT),
    ], axis=1).astype(np.float32)


def to_features(X: np.ndarray, mode: str = "both") -> np.ndarray:
    single = X.ndim == 1
    Xb = X.reshape(1, -1) if single else X
    assert Xb.shape[1] == FEATURE_DIM, Xb.shape
    if mode == "raw":
        out = Xb
    elif mode == "derived":
        # keep presence flags + derived (drop raw coords entirely)
        out = np.concatenate(
            [Xb[:, [LH_PRESENT, RH_PRESENT]], derived(Xb)], axis=1)
    elif mode == "both":
        out = np.concatenate([Xb, derived(Xb)], axis=1)
    elif mode == "body_raw_hands_derived":
        # raw body coords + presence flags + all derived (no raw hand coords)
        out = np.concatenate(
            [Xb[:, : N_BODY * 2], Xb[:, [LH_PRESENT, RH_PRESENT]], derived(Xb)], axis=1)
    else:
        raise ValueError(mode)
    return out[0] if single else out


def out_dim(mode: str) -> int:
    return {
        "raw": FEATURE_DIM,
        "derived": 2 + DERIVED_DIM,
        "both": FEATURE_DIM + DERIVED_DIM,
        "body_raw_hands_derived": N_BODY * 2 + 2 + DERIVED_DIM,
    }[mode]


def demo() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, FEATURE_DIM)).astype(np.float32)
    X[:, LH_PRESENT] = 1.0
    X[:, RH_PRESENT] = [1, 0, 1, 0, 1]
    for m in ("raw", "derived", "both", "body_raw_hands_derived"):
        Y = to_features(X, m)
        assert Y.shape == (5, out_dim(m)), (m, Y.shape, out_dim(m))
        assert np.isfinite(Y).all(), m
    # absent right hand -> its derived block is zero
    d = derived(X)
    assert np.allclose(d[1, N_BODY_DERIVED + N_HAND_DERIVED:], 0.0)
    assert not np.allclose(d[0, N_BODY_DERIVED + N_HAND_DERIVED:], 0.0)
    # scale invariance: real hands are wrist-centred; scaling about the wrist
    # must not change the angle features
    X2 = X.copy()
    lh = X2[:, LH_OFF: LH_OFF + N_HAND * 2].reshape(5, N_HAND, 2)
    lh[:] = (lh - lh[:, :1]) * 2.5 + lh[:, :1]
    X2[:, LH_OFF: LH_OFF + N_HAND * 2] = lh.reshape(5, -1)
    d2 = derived(X2)
    lh_block = slice(N_BODY_DERIVED, N_BODY_DERIVED + N_HAND_DERIVED)
    assert np.allclose(d[:, lh_block], d2[:, lh_block], atol=3e-3), \
        "hand derived features must be scale-free"
    print("derived.demo OK")


if __name__ == "__main__":
    demo()
