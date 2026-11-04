"""Turn cached keypoints (data/kpts) into frame-level train/val tables with binary labels.

    python -m pipeline.build_dataset

Label rules (all documented in docs/datasets.md; they only CLEAN weak labels, the classifier itself has no rules):
  * pos clips (K700 'waving hand', HMDB 'wave'): frame = 1 if a wrist is clearly above its shoulder, else dropped
    (a waving clip also contains frames with the hand down: those are unknown, not negative).
  * hard-negative / neutral clips (scratch-like actions, phone, drinking, reading ...): frame = 0, except frames where a
    wrist is far above the head (might be a genuine raise) -> dropped.
  * SCB: person box matched to GT; cls 0 (hand-raising) -> 1 if a wrist is above its shoulder else dropped;
    cls 1/2 (reading/writing) -> 0; unmatched persons -> dropped.
  * saluting ('ambig') is excluded.
Split: by source video group (YouTube id / movie name); SCB uses its own train/val split. The user's two clips
are never touched here (they are the test set, see classifier/evaluate.py).
Each row stores features of the original AND the mirrored skeleton (mirror augmentation, label unchanged).
"""
from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from features.keypoints import (FEATURE_NAMES, featurize, mirror, normalize, features_from_raw)

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
RISE_POS = 0.10        # wrist this far above shoulder (in shoulder-width units) counts as raised for positive labels
HIGH_ABOVE_NOSE = 0.55 # in negative clips, wrist this far above the nose = clearly above the head -> maybe a real raise
VAL_FRAC = 0.15
MIN_BOX_FRAC = 0.20    # videos: person box height / frame height
MIN_BOX_PX = 100       # SCB: person box height in px


def group_of(row) -> str:
    cid, lab = row["clip_id"], row["label"].replace(" ", "_")
    if row["source"] == "hmdb51":
        rest = cid[len(lab) + 1:] if cid.startswith(lab + "_") else cid
        return "hmdb:" + rest.split(f"_{lab}_")[0]
    return "k700:" + cid


def is_val(group: str) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 1000 < VAL_FRAC * 1000


def row_features(kpt, box):
    xy, c, ok = normalize(kpt, box)
    xm, cm = mirror(xy, c)
    return ok, featurize(xy, c, ok), featurize(xm, cm, ok)


def label_frame(role: str, f: np.ndarray):
    """-> 1 / 0 / None (drop) from the frame's features."""
    rise = f[IDX["best_wrist_rise"]]
    wr_vis = f[IDX["A_wr_sh_dy"]] != 0 or f[IDX["B_wr_sh_dy"]] != 0
    if role == "pos":
        return 1 if (wr_vis and rise > RISE_POS) else None
    high = f[IDX["A_wr_above_nose"]] > HIGH_ABOVE_NOSE or f[IDX["B_wr_above_nose"]] > HIGH_ABOVE_NOSE
    return None if high else 0


def main() -> None:
    rows = {k: [] for k in ("X", "Xm", "y", "source", "role", "label", "group", "split")}
    drop = {}

    def add(f, fm, y, source, role, label, group, split):
        for k, v in zip(rows, (f, fm, y, source, role, label, group, split)):
            rows[k].append(v)

    # ---- videos
    for src in ("k700", "hmdb51"):
        for r in csv.DictReader(open(C.RAW / src / "manifest.csv")):
            if r["role"] == "ambig":
                continue
            p = C.KPTS / r["source"] / f"{r['clip_id']}.npz"
            if not p.exists():
                drop["no_kpts"] = drop.get("no_kpts", 0) + 1; continue
            d = np.load(p); g = group_of(r); split = "val" if is_val(g) else "train"
            H = float(d["frame_h"])
            for kpt, box in zip(d["kpt"], d["box"]):
                if box[3] - box[1] < MIN_BOX_FRAC * H:
                    drop["small_or_missing_person"] = drop.get("small_or_missing_person", 0) + 1; continue
                ok, f, fm = row_features(kpt, box)
                if not ok:
                    drop["no_shoulders"] = drop.get("no_shoulders", 0) + 1; continue
                y = label_frame(r["role"], f)
                if y is None:
                    drop[f"cleaned_{r['role']}"] = drop.get(f"cleaned_{r['role']}", 0) + 1; continue
                add(f, fm, y, r["source"], r["role"], r["label"], g, split)

    # ---- SCB
    names = {0: ("pos", "hand-raising"), 1: ("hardneg", "reading"), 2: ("hardneg", "writing")}
    for split in ("train", "val"):
        p = C.KPTS / "scb" / f"{split}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        for kpt, box, cls, img in zip(d["kpt"], d["box"], d["scb_cls"], d["image"]):
            if cls < 0:
                drop["scb_unmatched"] = drop.get("scb_unmatched", 0) + 1; continue
            if box[3] - box[1] < MIN_BOX_PX:
                drop["scb_small"] = drop.get("scb_small", 0) + 1; continue
            ok, f, fm = row_features(kpt, box)
            if not ok:
                drop["no_shoulders"] = drop.get("no_shoulders", 0) + 1; continue
            role, lab = names[int(cls)]
            y = label_frame(role, f)
            if y is None:
                drop[f"cleaned_scb_{lab}"] = drop.get(f"cleaned_scb_{lab}", 0) + 1; continue
            add(f, fm, y, "scb", role, lab, f"scb:{split}:{str(img).split('_')[0]}", split)

    C.DATASET.mkdir(parents=True, exist_ok=True)
    arr = {k: np.array(v) for k, v in rows.items()}
    for split in ("train", "val"):
        m = arr["split"] == split
        np.savez_compressed(C.DATASET / f"{split}.npz", **{k: v[m] for k, v in arr.items() if k != "split"},
                            feature_names=np.array(FEATURE_NAMES))
    print("dropped:", drop)
    print(f"{'split':6s} {'source':10s} {'role':8s} {'label':26s} {'n':>7s} {'pos%':>6s}")
    keys = sorted({(s, so, ro, la) for s, so, ro, la in zip(arr["split"], arr["source"], arr["role"], arr["label"])})
    agg = {}
    for s, so, ro, la in keys:
        m = (arr["split"] == s) & (arr["source"] == so) & (arr["role"] == ro) & (arr["label"] == la)
        agg[(s, so, ro)] = [agg.get((s, so, ro), [0, 0])[0] + int(m.sum()), agg.get((s, so, ro), [0, 0])[1] + int(arr["y"][m].sum())]
    for (s, so, ro), (n, p) in sorted(agg.items()):
        print(f"{s:6s} {so:10s} {ro:8s} {'(all classes)':26s} {n:7d} {p / n:6.0%}")
    for s in ("train", "val"):
        m = arr["split"] == s
        print(f"{s}: {int(m.sum())} frames, {int(arr['y'][m].sum())} positive, {len(set(arr['group'][m]))} groups")


if __name__ == "__main__":
    main()
