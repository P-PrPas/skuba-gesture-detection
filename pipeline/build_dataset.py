"""Phase 2b: build the hybrid train/test dataset.

    python -m pipeline.build_dataset

TRAIN comes from external datasets (data/features_ext/), cleaned + augmented.
TEST is every original s01/s02 frame (data/features/), never augmented — a
genuine cross-domain / cross-subject held-out set.

`glico_pose` has no external data (and `sit`/`squat`/`laying` fall back to this
if the COCO posture mine comes up empty): train = augmented copies of the
s01/s02 frames, the originals still go to test — a leaky number, flagged
`aug_only` in the card. `thumb` has no s01 data, so 15% of its external rows are
held out as its test.

Writes data/dataset/{train,test}.npz + dataset_card.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from features.augment import AugParams, augment_once
from features.schema import CLASSES, FEATURE_DIM, class_index

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"          # our s01/s02, per clip
FEAT_EXT = ROOT / "data" / "features_ext"  # external, per (source,class)
OUT = ROOT / "data" / "dataset"
SEED = 20260831

# i_love_you, rock, heart were CUT from CLASSES on 2026-09-01 (see CLAUDE.md).
# The `for cls in CLASSES` loop below never sees them; their leftover
# data/features_ext/*.npz (roboflow_ily, hagrid__rock) load but are ignored.
PENDING_DATA: set[str] = set()

# external data is chest-level but the class is performed overhead -> expand each
# external row into raised-arm variants before the normal augmentation.
ARMS_UP = {"mini_heart"}

# fallback only — used when a class has NO external source (the external path
# below wins whenever coco_pose__<cls>.npz etc. exist). sit + laying are mined
# from COCO now; squat's COCO labels overlapped sit too much (see classifier/
# train.py) so squat stays here on aug(s01); glico_pose has no dataset.
AUG_ONLY = {"sit", "laying", "squat", "glico_pose"}
# aux external labels that are really "not a target gesture"
AUX_TO_CLASS = {"_ily_negative": "idle", "_coco_idle": "idle"}
HANDSHAPE = {"ok", "thumb", "two_finger", "mini_heart", "_ily_negative"}
THUMB_TEST_FRAC = 0.15

# keep the class balance sane: subsample external originals to this before
# augmenting; grow aug-only classes toward this many rows with extra copies.
EXT_ORIG_CAP = 1400
AUG_ONLY_TARGET = 2200

# MediaPipe-33 body indices
_NOSE, _LSHO, _RSHO, _LWRI, _RWRI = 0, 11, 12, 15, 16


def _body(X):
    return X[:, : FEATURE_DIM].reshape(len(X), -1, 2)[:, :33]


def _clean_external(X: np.ndarray, cls: str) -> np.ndarray:
    """Boolean keep-mask. Drops blown normalisation, missing hands on hand-shape
    classes, and COCO pose rows whose MediaPipe keypoints don't actually show the
    pose (COCO's own annotation often points at a different person in the frame)."""
    keep = ~(np.abs(X[:, :151]) > 12).any(axis=1)
    b = _body(X)
    if cls in HANDSHAPE:
        keep &= (X[:, 108] > 0.5) | (X[:, 151] > 0.5)
    elif cls == "raise_right_hand":
        keep &= b[:, _RWRI, 1] < b[:, _NOSE, 1] - 0.1
    elif cls == "raise_left_hand":
        keep &= b[:, _LWRI, 1] < b[:, _NOSE, 1] - 0.1
    elif cls == "t_pose":
        sho_y = (b[:, _LSHO, 1] + b[:, _RSHO, 1]) / 2
        sho_w = np.abs(b[:, _LSHO, 0] - b[:, _RSHO, 0]) + 1e-6
        keep &= (np.abs(b[:, _LWRI, 1] - sho_y) < 0.5) & (np.abs(b[:, _RWRI, 1] - sho_y) < 0.5)
        keep &= np.abs(b[:, _LWRI, 0] - b[:, _RWRI, 0]) > 2.0 * sho_w
    elif cls in ("sit", "squat", "laying"):
        # re-verify the COCO 2D auto-label against MediaPipe's normalized body
        # (units ~= shoulder-hip distance). 11/12 sho, 23/24 hip, 25/26 knee.
        sho_m = (b[:, 11] + b[:, 12]) / 2
        hip_m = (b[:, 23] + b[:, 24]) / 2
        knee_dy = (b[:, 25, 1] + b[:, 26, 1]) / 2 - hip_m[:, 1]      # + = knee below hip
        spine_h = np.abs(sho_m[:, 0] - hip_m[:, 0]) > np.abs(sho_m[:, 1] - hip_m[:, 1])
        if cls == "laying":
            keep &= spine_h
        else:
            keep &= ~spine_h & (sho_m[:, 1] < hip_m[:, 1])          # upright torso
            keep &= knee_dy < (0.5 if cls == "squat" else 1.2)      # hips low / thighs off-vertical
    return keep


def _load_external():
    """-> {class: (X, subject_ids, source)} after cleaning + aux remap."""
    out: dict[str, list] = defaultdict(lambda: [[], [], []])
    stats = {}
    for f in sorted(FEAT_EXT.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        raw_cls = str(d["label"][0])
        cls = AUX_TO_CLASS.get(raw_cls, raw_cls)
        X = d["X"].astype(np.float32)
        mask = _clean_external(X, raw_cls)
        stats[f.name] = (len(X), int(mask.sum()))
        out[cls][0].append(X[mask])
        out[cls][1].append(d["subject_id"][mask])
        out[cls][2].append(np.array([str(d["source"][0])] * int(mask.sum())))
    merged = {}
    for cls, (xs, sids, srcs) in out.items():
        merged[cls] = (np.concatenate(xs), np.concatenate(sids), np.concatenate(srcs))
    return merged, stats


def _load_ours():
    """-> {class: (X, subject, session, clip)} from our s01/s02 per-clip files."""
    out: dict[str, list] = defaultdict(lambda: [[], [], [], []])
    for f in sorted(FEAT.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        cls = str(d["label"][0])
        out[cls][0].append(d["X"].astype(np.float32))
        out[cls][1].append(d["subject_id"])
        out[cls][2].append(d["session_id"])
        out[cls][3].append(d["clip_id"])
    return {c: tuple(np.concatenate(p) for p in parts) for c, parts in out.items()}


def _augment_rows(X, labels, p: AugParams, seed_off: int, n_copies: int, keep_orig: bool,
                  elevate: bool = False):
    rng = np.random.default_rng(SEED + seed_off)
    ax, ay = [], []
    for i in range(len(X)):
        if keep_orig:
            ax.append(X[i]); ay.append(labels[i])
        for _ in range(n_copies):
            v, lab = augment_once(X[i], str(labels[i]), p, rng, elevate=elevate)
            ax.append(v); ay.append(lab)
    return np.stack(ax).astype(np.float32), np.array(ay)


def _write(name, X, y_str, extra: dict):
    y = np.array([class_index(str(l)) for l in y_str], dtype=np.int64)
    np.savez_compressed(OUT / f"{name}.npz", X=X.astype(np.float32), y=y, label=y_str, **extra)
    counts = {c: int((y == class_index(c)).sum()) for c in CLASSES if (y == class_index(c)).any()}
    return {"rows": int(len(X)), "class_counts": counts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.npz"):
        old.unlink()
    p = AugParams()
    ext, clean_stats = _load_external()
    ours = _load_ours()

    tr_X, tr_y = [], []
    te_X, te_y, te_meta_src, te_subj = [], [], [], []
    per_class = {}
    rng = np.random.default_rng(SEED)

    for cls in CLASSES:
        rec = {"train_source": None, "train_rows": 0, "test_source": None,
               "test_rows": 0, "eval_type": None}

        # ---- TRAIN ----
        if cls in PENDING_DATA:
            rec["eval_type"] = "pending_data"      # in test.npz only, never trained
        elif cls in ext:                           # external data -> real cross-domain
            # (AUG_ONLY is only the fallback for classes with no external source)
            Xe, _, _ = ext[cls]
            if cls == "thumb":
                idx = rng.permutation(len(Xe))
                n_te = int(len(Xe) * THUMB_TEST_FRAC)
                te_idx, tr_idx = idx[:n_te], idx[n_te:]
                Xtr = Xe[tr_idx]
                te_X.append(Xe[te_idx]); te_y.append(np.array(["thumb"] * n_te))
                te_meta_src.append(np.array(["hagrid_heldout"] * n_te))
                te_subj.append(np.array(["hagrid"] * n_te))
                rec["test_source"] = "held-out HaGRID (15%)"
                rec["test_rows"] = n_te
                rec["eval_type"] = "held_out_external"
            else:
                Xtr = Xe
            if len(Xtr) > EXT_ORIG_CAP:  # keep class balance sane
                Xtr = Xtr[rng.permutation(len(Xtr))[:EXT_ORIG_CAP]]
            # scarce external classes (t_pose) get more copies to reach ~AUG_ONLY_TARGET
            nc = max(p.n_per_sample, min(40, -(-AUG_ONLY_TARGET // max(1, len(Xtr)))))
            up = cls in ARMS_UP
            aX, aY = _augment_rows(Xtr, np.array([cls] * len(Xtr)), p,
                                   seed_off=class_index(cls), n_copies=nc, keep_orig=not up,
                                   elevate=up)
            tr_X.append(aX); tr_y.append(aY)
            rec["train_source"] = ("external+arm-elevation-aug: " if up else "external: ") + \
                "/".join(sorted(set(ext[cls][2].tolist())))
            rec["train_rows"] = int(len(aX))
        elif cls in AUG_ONLY and cls in ours:
            Xo_all, subj_all = ours[cls][0], ours[cls][1]
            # augment from s01 only when possible, so any s02 frames stay a clean
            # cross-person test; if s01 has none (laying), fall back to all (full leak).
            s01m = subj_all == "s01"
            Xo = Xo_all[s01m] if s01m.any() else Xo_all
            train_subj = "s01" if s01m.any() else "s02"
            n_copies = min(40, max(4, -(-AUG_ONLY_TARGET // max(1, len(Xo)))))
            aX, aY = _augment_rows(Xo, np.array([cls] * len(Xo)), p,
                                   seed_off=class_index(cls), n_copies=n_copies, keep_orig=False)
            tr_X.append(aX); tr_y.append(aY)
            rec["train_source"] = f"aug({train_subj}) only x{n_copies}"
            rec["train_rows"] = int(len(aX))
            rec["eval_type"] = "aug_only_leak"

        # ---- TEST ----  (all original s01/s02 frames for this class)
        if cls in ours:
            Xo, subj = ours[cls][0], ours[cls][1]
            te_X.append(Xo); te_y.append(np.array([cls] * len(Xo)))
            te_meta_src.append(np.array(["s01/s02"] * len(Xo)))
            te_subj.append(subj)
            rec["test_source"] = (rec["test_source"] + " + s01/s02") if rec["test_source"] else "s01/s02"
            rec["test_rows"] += len(Xo)
            rec["test_subjects"] = sorted(set(subj.tolist()))
            if rec["eval_type"] is None:
                rec["eval_type"] = "cross_domain" if cls in ext else "n/a"

        per_class[cls] = rec

    trX = np.concatenate(tr_X); trY = np.concatenate(tr_y)
    teX = np.concatenate(te_X); teY = np.concatenate(te_y)
    teSRC = np.concatenate(te_meta_src); teSUBJ = np.concatenate(te_subj)

    # shuffle train
    sh = np.random.default_rng(SEED + 7).permutation(len(trX))
    trX, trY = trX[sh], trY[sh]

    card = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_dim": FEATURE_DIM,
        "classes": CLASSES,
        "strategy": "train=external(+aug); test=original s01/s02 (+held-out HaGRID for thumb)",
        "augmentation": {"applied_to": "train only", **p.__dict__},
        "external_clean": {k: {"raw": v[0], "kept": v[1]} for k, v in clean_stats.items()},
        "train": _write("train", trX, trY, {}),
        "test": _write("test", teX, teY, {"source": teSRC, "subject_id": teSUBJ}),
        "per_class": per_class,
        "notes": [
            "cross_domain classes: real generalisation number (external train, our test).",
            "held_out_external (thumb): cross-subject within HaGRID, same webcam domain.",
            "aug_only_leak classes: train is augmented copies of s01 frames. For `sit` "
            "the s02 test frames ARE a clean cross-person check (aug is s01-only); for "
            "`laying` there is no s01 so train and test are the same s02 frames (full "
            "leak). Phase 6 field test is the real gate for all aug_only classes.",
        ],
    }
    (OUT / "dataset_card.json").write_text(json.dumps(card, indent=2, default=str))

    print(f"train {card['train']['rows']} rows, test {card['test']['rows']} rows -> {OUT}\n")
    print(f"{'class':18s} {'eval':20s} {'train':>7s} {'test':>6s}")
    for c in CLASSES:
        r = per_class[c]
        print(f"{c:18s} {str(r['eval_type'] or '-'):20s} {r['train_rows']:7d} {r['test_rows']:6d}")


if __name__ == "__main__":
    main()
