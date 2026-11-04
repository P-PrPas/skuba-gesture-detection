"""Keypoint -> feature contract shared by training and the realtime detector.

Pipeline: raw COCO-17 (x, y, conf) + person box
  -> normalize()  : person-centred, scale-invariant, unseen joints zeroed
  -> mirror()     : horizontal flip augmentation (class is symmetric: label unchanged)
  -> featurize()  : fixed-length vector (raw upper-body coords + conf + derived geometry)

Everything is coordinate-based; no thresholds on "raised" live here (the classifier learns that).
"""
from __future__ import annotations
import numpy as np
from config import KPT_VIS

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = range(13)
MIRROR_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
N_UPPER = 13  # nose..hips used as raw features
EPS = 1e-6


def normalize(kpt: np.ndarray, box: np.ndarray):
    """kpt [17,3] (x,y,conf), box [4] xyxy -> (xy_n [17,2], conf [17], shoulders_ok)."""
    xy, c = kpt[:, :2].astype(np.float32), kpt[:, 2].astype(np.float32)
    vis = c >= KPT_VIS
    sh_ok = bool(vis[L_SH] and vis[R_SH])
    bw, bh = max(float(box[2] - box[0]), 1.0), max(float(box[3] - box[1]), 1.0)
    if sh_ok:
        origin = (xy[L_SH] + xy[R_SH]) / 2
        scale = float(np.linalg.norm(xy[L_SH] - xy[R_SH]))
        if vis[NOSE]:
            scale = max(scale, 2 * float(np.linalg.norm(xy[NOSE] - origin)))
    else:  # fall back to the person box (top-centre-ish)
        origin = np.array([(box[0] + box[2]) / 2, box[1] + 0.2 * bh], np.float32)
        scale = 0.5 * bw
    scale = max(scale, 1.0)
    out = (xy - origin) / scale
    out[~vis] = 0.0
    return out.astype(np.float32), np.where(vis, c, 0.0).astype(np.float32), sh_ok


def mirror(xy_n: np.ndarray, c: np.ndarray):
    """Horizontal flip: swap left/right joints, negate x. Label is unchanged."""
    xy = xy_n[MIRROR_PERM].copy(); xy[:, 0] *= -1
    return xy, c[MIRROR_PERM].copy()


def _arm(xy, c, sh, el, wr):
    """14 derived features for one arm (zeros when the needed joints are unseen)."""
    f = np.zeros(14, np.float32)
    ok_w, ok_e, ok_s = c[wr] > 0, c[el] > 0, c[sh] > 0
    if ok_w and ok_s:
        d = xy[wr] - xy[sh]; f[0:2] = d
    if ok_w and ok_e:
        f[2:4] = xy[wr] - xy[el]
    if ok_e and ok_s:
        f[4:6] = xy[el] - xy[sh]
    if ok_w and ok_e and ok_s:
        u, v = xy[sh] - xy[el], xy[wr] - xy[el]
        f[6] = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + EPS))  # elbow flexion cosine
    if ok_e and ok_s:
        f[7] = np.linalg.norm(xy[el] - xy[sh])
    if ok_w and ok_e:
        f[8] = np.linalg.norm(xy[wr] - xy[el])
    if ok_w and c[NOSE] > 0:
        f[9] = np.linalg.norm(xy[wr] - xy[NOSE])
    ears = [e for e in (L_EAR, R_EAR) if c[e] > 0]
    if ok_w and ears:
        f[10] = min(np.linalg.norm(xy[wr] - xy[e]) for e in ears)
    if ok_w and c[L_EYE] > 0 and c[R_EYE] > 0:
        f[11] = np.linalg.norm(xy[wr] - (xy[L_EYE] + xy[R_EYE]) / 2)
    if ok_w and c[NOSE] > 0:
        f[12] = xy[NOSE, 1] - xy[wr, 1]          # >0: wrist above nose
    if ok_e and ok_s:
        f[13] = xy[sh, 1] - xy[el, 1]            # >0: elbow above shoulder
    return f


def featurize(xy_n: np.ndarray, c: np.ndarray, sh_ok: bool) -> np.ndarray:
    upper = xy_n[:N_UPPER]
    la = _arm(xy_n, c, L_SH, L_EL, L_WR)
    ra = _arm(xy_n, c, R_SH, R_EL, R_WR)
    wr_rise = [(xy_n[s, 1] - xy_n[w, 1]) for w, s in ((L_WR, L_SH), (R_WR, R_SH))
               if c[w] > 0 and c[s] > 0]
    wr_nose = [np.linalg.norm(xy_n[w] - xy_n[NOSE]) for w in (L_WR, R_WR) if c[w] > 0 and c[NOSE] > 0]
    el_up = [xy_n[s, 1] - xy_n[e, 1] for e, s in ((L_EL, L_SH), (R_EL, R_SH)) if c[e] > 0 and c[s] > 0]
    wr_wr = np.linalg.norm(xy_n[L_WR] - xy_n[R_WR]) if c[L_WR] > 0 and c[R_WR] > 0 else 0.0
    agg = np.array([max(wr_rise) if wr_rise else 0.0,
                    min(wr_nose) if wr_nose else 0.0,
                    max(el_up) if el_up else 0.0,
                    wr_wr, float(sh_ok), float((c[:N_UPPER] > 0).mean())], np.float32)
    # Symmetric (left/right-agnostic) ordering of per-arm features keeps mirror-equivariance:
    # arm A = arm whose wrist is higher, so mirroring changes nothing.
    first, second = (la, ra) if la[0 + 1] <= ra[0 + 1] else (ra, la)
    return np.concatenate([upper.ravel(), c[:N_UPPER], first, second, agg]).astype(np.float32)


def _names():
    base = ["nose", "leye", "reye", "lear", "rear", "lsh", "rsh", "lel", "rel", "lwr", "rwr", "lhip", "rhip"]
    n = [f"{b}_{a}" for b in base for a in "xy"] + [f"{b}_conf" for b in base]
    arm = ["wr_sh_dx", "wr_sh_dy", "wr_el_dx", "wr_el_dy", "el_sh_dx", "el_sh_dy", "elbow_cos",
           "upper_len", "fore_len", "wr_nose", "wr_ear", "wr_eyes", "wr_above_nose", "el_above_sh"]
    n += [f"A_{a}" for a in arm] + [f"B_{a}" for a in arm]
    n += ["best_wrist_rise", "min_wr_nose", "max_el_above_sh", "wr_wr", "shoulders_ok", "frac_visible"]
    return n


FEATURE_NAMES = _names()
FEATURE_DIM = len(FEATURE_NAMES)
RAW_DIM = N_UPPER * 3           # first block: raw xy + conf only (for the ablation)
DERIVED_SLICE = slice(RAW_DIM, FEATURE_DIM)


def features_from_raw(kpt: np.ndarray, box: np.ndarray) -> np.ndarray:
    xy, c, ok = normalize(kpt, box)
    return featurize(xy, c, ok)


def rule_raised(kpt: np.ndarray, vis: float = KPT_VIS) -> bool:
    """Baseline only (diagnosis / comparison): a wrist is above its same-side shoulder."""
    return any(kpt[w, 2] > vis and kpt[s, 2] > vis and kpt[w, 1] < kpt[s, 1]
               for w, s in ((9, 5), (10, 6)))
