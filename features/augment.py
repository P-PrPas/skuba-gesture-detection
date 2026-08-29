"""Augmentation (ARCHITECTURE.md "Augmentation recipe"). Operates on the fused
152-vector + string label. Starting parameters live in AugParams - tune in
Phase 4 and record the final values in the dataset card.

HARD CONSTRAINT (CLAUDE.md #2): mirror must flip x, swap the left/right hand
slices, swap left/right body landmarks, AND relabel left/right classes.
`demo()` at the bottom checks this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (
    FEATURE_DIM,
    LH_OFF,
    LH_PRESENT,
    MIRROR_LABEL_SWAP,
    N_BODY,
    N_HAND,
    POSE_LR_PAIRS,
    RH_OFF,
    RH_PRESENT,
)


@dataclass
class AugParams:
    mirror_p: float = 0.5
    rot_deg: float = 12.0          # max |camera tilt| simulated
    kp_dropout_frac: float = 0.05  # fraction of keypoints zeroed (occlusion)
    hand_drop_p: float = 0.10      # chance a present hand is dropped entirely
    coord_noise_std: float = 0.02  # gaussian, in normalized units
    n_per_sample: int = 4          # augmented copies per original (train only)


# ---- slice helpers ----
_BODY = slice(0, N_BODY * 2)
_LH = slice(LH_OFF, LH_OFF + N_HAND * 2)
_RH = slice(RH_OFF, RH_OFF + N_HAND * 2)


def _xy(v: np.ndarray, sl: slice) -> np.ndarray:
    return v[sl].reshape(-1, 2)


def _set_xy(v: np.ndarray, sl: slice, xy: np.ndarray) -> None:
    v[sl] = xy.reshape(-1)


def mirror(v: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    out = v.copy()
    # body: flip x, then swap L/R landmark pairs
    b = _xy(out, _BODY)
    b[:, 0] *= -1.0
    for i, j in POSE_LR_PAIRS:
        b[[i, j]] = b[[j, i]]
    _set_xy(out, _BODY, b)

    # hands: mirrored right hand becomes the left hand and vice-versa
    lh, rh = _xy(v, _LH).copy(), _xy(v, _RH).copy()
    lh[:, 0] *= -1.0
    rh[:, 0] *= -1.0
    _set_xy(out, _LH, rh)
    _set_xy(out, _RH, lh)
    out[LH_PRESENT], out[RH_PRESENT] = v[RH_PRESENT], v[LH_PRESENT]

    return out, MIRROR_LABEL_SWAP.get(label, label)


def _rotate(v: np.ndarray, deg: float) -> np.ndarray:
    out = v.copy()
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    for sl in (_BODY, _LH, _RH):
        _set_xy(out, sl, _xy(out, sl) @ R.T)
    return out


def _keypoint_dropout(v: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = v.copy()
    total = N_BODY + 2 * N_HAND
    k = int(round(frac * total))
    if k <= 0:
        return out
    idx = rng.choice(total, size=k, replace=False)
    for gi in idx:
        if gi < N_BODY:
            out[2 * gi : 2 * gi + 2] = 0.0
        elif gi < N_BODY + N_HAND:
            h = gi - N_BODY
            out[LH_OFF + 2 * h : LH_OFF + 2 * h + 2] = 0.0
        else:
            h = gi - N_BODY - N_HAND
            out[RH_OFF + 2 * h : RH_OFF + 2 * h + 2] = 0.0
    return out


def _hand_drop(v: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    out = v.copy()
    if out[LH_PRESENT] and rng.random() < p:
        out[_LH] = 0.0
        out[LH_PRESENT] = 0.0
    if out[RH_PRESENT] and rng.random() < p:
        out[_RH] = 0.0
        out[RH_PRESENT] = 0.0
    return out


def _coord_noise(v: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    out = v.copy()
    for sl in (_BODY, _LH, _RH):
        out[sl] += rng.normal(0.0, std, size=out[sl].shape).astype(np.float32)
    return out


def augment_once(
    v: np.ndarray, label: str, p: AugParams, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    out, lab = (mirror(v, label) if rng.random() < p.mirror_p else (v.copy(), label))
    out = _rotate(out, rng.uniform(-p.rot_deg, p.rot_deg))
    out = _hand_drop(out, p.hand_drop_p, rng)
    out = _keypoint_dropout(out, p.kp_dropout_frac, rng)
    out = _coord_noise(out, p.coord_noise_std, rng)
    return out.astype(np.float32), lab


def demo() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, FEATURE_DIM).astype(np.float32)
    v[LH_PRESENT] = 1.0
    v[RH_PRESENT] = 0.0
    v[_RH] = 0.0

    m, lab = mirror(v, "raise_right_hand")
    assert lab == "raise_left_hand", lab
    # left slice of mirror == x-flipped right slice of original (which was zeros)
    assert np.allclose(m[_LH], 0.0) and m[LH_PRESENT] == 0.0
    # right slice of mirror == x-flipped original left hand
    lh = v[_LH].reshape(-1, 2).copy()
    lh[:, 0] *= -1
    assert np.allclose(m[_RH], lh.reshape(-1)) and m[RH_PRESENT] == 1.0
    # double mirror is identity (body + hands + label)
    mm, lab2 = mirror(m, lab)
    assert lab2 == "raise_right_hand"
    assert np.allclose(mm, v, atol=1e-6)
    # non-paired class keeps its label
    _, lab3 = mirror(v, "sit")
    assert lab3 == "sit"
    # rotation by 0 is identity
    assert np.allclose(_rotate(v, 0.0), v, atol=1e-6)
    print("augment.demo OK")


if __name__ == "__main__":
    demo()
