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
    POSE_LEFT_HIP,
    POSE_LEFT_SHOULDER,
    POSE_LR_PAIRS,
    POSE_RIGHT_HIP,
    POSE_RIGHT_SHOULDER,
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
    limb_jitter: float = 0.15      # max +/- limb-length scale — fakes body-proportion variety
    n_per_sample: int = 4          # augmented copies per original (train only)
    elevate_dy: tuple[float, float] = (-1.1, -0.3)  # arm-raise shift (norm units), opt-in per class


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


# ---- limb-length jitter: fake different body proportions ----
# Kinematic chains in the MediaPipe-33 body. Each chain is scaled about its
# first joint by one random factor; landmarks hanging off a chain tip (hands,
# feet) are translated rigidly with that tip. The pelvis (hip midpoint) is the
# fixed root, so the normalization reference (shoulder-hip) is barely disturbed.
_SL, _SR = POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER
_HL, _HR = POSE_LEFT_HIP, POSE_RIGHT_HIP
_CHAINS = {
    "arm": ([_SL, 13, 15], [_SR, 14, 16]),   # shoulder -> elbow -> wrist
    "leg": ([_HL, 25, 27], [_HR, 26, 28]),   # hip -> knee -> ankle
}
_TIP_FOLLOWS = {15: (17, 19, 21), 16: (18, 20, 22),   # hand pts follow the wrist
                27: (29, 31), 28: (30, 32)}           # foot pts follow the ankle
_FACE = tuple(range(0, 11))                            # nose..mouth follow the shoulder midpoint


def _limb_jitter(v: np.ndarray, amt: float, rng: np.random.Generator) -> np.ndarray:
    if amt <= 0:
        return v.copy()
    out = v.copy()
    b = _xy(out, _BODY)
    f = {k: float(rng.uniform(1 - amt, 1 + amt)) for k in ("arm", "leg", "torso")}

    # torso: scale shoulders about the hip midpoint; head rides with the shoulder mid
    hip_mid = (b[_HL] + b[_HR]) / 2
    sho_mid_old = (b[_SL] + b[_SR]) / 2
    for i in (_SL, _SR):
        b[i] = hip_mid + f["torso"] * (b[i] - hip_mid)
    d_head = (b[_SL] + b[_SR]) / 2 - sho_mid_old
    for i in _FACE:
        b[i] = b[i] + d_head

    # arms / legs: walk each chain from its (already-updated) root joint outward
    for group in ("arm", "leg"):
        for chain in _CHAINS[group]:
            for parent, child in zip(chain, chain[1:]):
                new_child = b[parent] + f[group] * (b[child] - b[parent])
                if child in _TIP_FOLLOWS:
                    d = new_child - b[child]
                    for t in _TIP_FOLLOWS[child]:
                        b[t] = b[t] + d
                b[child] = new_child

    _set_xy(out, _BODY, b)
    return out


_ARM_PTS = (13, 15, 17, 19, 21, 14, 16, 18, 20, 22)  # elbow, wrist, hand pts — both arms


def raise_arms(v: np.ndarray, dy: float) -> np.ndarray:
    """Shift both forearms + hand points vertically by `dy` (normalised body
    units, negative = up) so a chest-level two-hand gesture becomes overhead. A
    shared `dy` keeps the two hands at the same separation, which is the point for
    mini_heart / heart. Hand slices (_LH/_RH) are wrist-local so the handshape is
    untouched (docs/external_datasets.md R2.4). ponytail: rigid vertical shift,
    not a shoulder pivot — the upper arm stretches a little; fine for aug.
    """
    out = v.copy()
    b = _xy(out, _BODY)
    b[list(_ARM_PTS), 1] += dy
    _set_xy(out, _BODY, b)
    return out


def augment_once(
    v: np.ndarray, label: str, p: AugParams, rng: np.random.Generator, elevate: bool = False
) -> tuple[np.ndarray, str]:
    out, lab = (mirror(v, label) if rng.random() < p.mirror_p else (v.copy(), label))
    if elevate:
        out = raise_arms(out, rng.uniform(*p.elevate_dy))
    out = _limb_jitter(out, p.limb_jitter, rng)
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

    # limb jitter: amt 0 is identity; amt>0 moves a wrist but keeps the hip midpoint
    rng2 = np.random.default_rng(3)
    vb = rng2.normal(0, 1, FEATURE_DIM).astype(np.float32)
    assert np.allclose(_limb_jitter(vb, 0.0, rng2), vb)
    j = _limb_jitter(vb, 0.15, np.random.default_rng(1))
    bj, b0 = _xy(j, _BODY), _xy(vb, _BODY)
    hip0 = (b0[POSE_LEFT_HIP] + b0[POSE_RIGHT_HIP]) / 2
    hipj = (bj[POSE_LEFT_HIP] + bj[POSE_RIGHT_HIP]) / 2
    assert np.allclose(hip0, hipj, atol=1e-6), "hip root must not move"
    assert not np.allclose(bj[15], b0[15]), "a wrist should have moved"
    assert np.allclose(j[_LH], vb[_LH]) and np.allclose(j[_RH], vb[_RH]), "hand slices untouched"

    # raise_arms: wrists rise by dy, both hands move together (separation kept),
    # shoulders and hand slices untouched
    vr = np.zeros(FEATURE_DIM, np.float32)
    br = _xy(vr, _BODY)
    br[_SL] = [0.2, 0.0]; br[13] = [0.3, 0.4]; br[15] = [0.35, 0.8]
    br[_SR] = [-0.2, 0.0]; br[14] = [-0.3, 0.4]; br[16] = [-0.25, 0.8]
    _set_xy(vr, _BODY, br)
    vr[_LH] = np.arange(N_HAND * 2); vr[LH_PRESENT] = 1.0
    r = raise_arms(vr, -0.6)
    rb = _xy(r, _BODY)
    assert np.isclose(rb[15, 1], 0.2) and np.isclose(rb[16, 1], 0.2), "wrists shift by dy"
    gap0 = br[15, 0] - br[16, 0]; gap1 = rb[15, 0] - rb[16, 0]
    assert np.isclose(gap0, gap1), "hand separation preserved"
    assert np.allclose(rb[_SL], br[_SL]) and np.allclose(rb[_SR], br[_SR]), "shoulders fixed"
    assert np.allclose(r[_LH], vr[_LH]), "hand slice untouched by raise_arms"
    print("augment.demo OK")


if __name__ == "__main__":
    demo()
