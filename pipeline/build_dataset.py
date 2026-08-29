"""Phase 2 step 2: per-clip feature files -> versioned train/val/test dataset.

    python -m pipeline.build_dataset

- concatenates data/features/*.npz
- splits BY SUBJECT (CLAUDE.md #5). With <3 subjects it cannot make a valid
  held-out split: everything goes to train, val/test are empty, and the card
  is stamped phase2_exit_met=false.
- augments TRAIN ONLY (mirror+relabel, rotation, dropout, noise - see
  features/augment.py). Originals are kept.
- writes data/dataset/{train,val,test}.npz + dataset_card.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from features.augment import AugParams, augment_once
from features.schema import CLASSES, FEATURE_DIM, class_index

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "data" / "dataset"
SEED = 20260829
VAL_FRAC, TEST_FRAC = 0.15, 0.15


def _load_all() -> dict:
    files = sorted(p for p in FEAT_DIR.glob("*.npz"))
    if not files:
        raise SystemExit("no data/features/*.npz - run pipeline.extract_features first")
    parts = {k: [] for k in ("X", "label", "subject_id", "session_id", "clip_id", "frame_idx")}
    for p in files:
        d = np.load(p, allow_pickle=False)
        for k in parts:
            parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _split_subjects(subjects: np.ndarray) -> tuple[set, set, set]:
    uniq = sorted(set(subjects.tolist()))
    if len(uniq) < 3:
        return set(uniq), set(), set()
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_test = max(1, round(len(uniq) * TEST_FRAC))
    n_val = max(1, round(len(uniq) * VAL_FRAC))
    return set(uniq[n_val + n_test:]), set(uniq[:n_val]), set(uniq[n_val:n_val + n_test])


def _augment_train(X, y_str, subj, sess, clip, frame, p: AugParams):
    rng = np.random.default_rng(SEED + 1)
    Xa, ya, sa, ea, ca, fa, aug = [], [], [], [], [], [], []
    for i in range(len(X)):
        # keep original
        Xa.append(X[i]); ya.append(y_str[i]); sa.append(subj[i]); ea.append(sess[i])
        ca.append(clip[i]); fa.append(frame[i]); aug.append(False)
        for _ in range(p.n_per_sample):
            v, lab = augment_once(X[i], str(y_str[i]), p, rng)
            Xa.append(v); ya.append(lab); sa.append(subj[i]); ea.append(sess[i])
            ca.append(clip[i]); fa.append(frame[i]); aug.append(True)
    return (
        np.stack(Xa).astype(np.float32), np.array(ya), np.array(sa), np.array(ea),
        np.array(ca), np.array(fa, dtype=np.int32), np.array(aug),
    )


def _write(name: str, X, y_str, subj, sess, clip, frame, aug):
    y = np.array([class_index(str(l)) for l in y_str], dtype=np.int64) if len(y_str) else np.zeros(0, np.int64)
    np.savez_compressed(
        OUT_DIR / f"{name}.npz",
        X=X.astype(np.float32), y=y, label=y_str,
        subject_id=subj, session_id=sess, clip_id=clip, frame_idx=frame, is_augmented=aug,
    )
    counts = {c: int((y == class_index(c)).sum()) for c in CLASSES if (y == class_index(c)).any()}
    return {"rows": int(len(X)), "subjects": sorted(set(subj.tolist())), "class_counts": counts}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_all()
    assert data["X"].shape[1] == FEATURE_DIM, data["X"].shape

    tr, va, te = _split_subjects(data["subject_id"])
    valid_eval = bool(va and te)
    print(f"subjects -> train={sorted(tr)} val={sorted(va)} test={sorted(te)}")
    if not valid_eval:
        print("WARNING: <3 subjects - no valid held-out split. val/test empty, "
              "everything in train. Phase 2 exit criterion NOT met.")

    masks = {
        "train": np.isin(data["subject_id"], list(tr)),
        "val": np.isin(data["subject_id"], list(va)),
        "test": np.isin(data["subject_id"], list(te)),
    }
    p = AugParams()
    card = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pose_backbone": json.loads((FEAT_DIR / "_extract_summary.json").read_text())["pose_backbone"],
        "feature_dim": FEATURE_DIM,
        "feature_layout": "body[0:66] lh[66:108] lh_present[108] rh[109:151] rh_present[151]",
        "classes": CLASSES,
        "split_method": "by_subject" if valid_eval else "all_to_train (single/low subject count)",
        "phase2_exit_met": valid_eval,
        "phase2_exit_reason": "" if valid_eval else f"only {len(tr | va | te)} subject(s); need >=3 for a subject-wise held-out split",
        "augmentation": {"applied_to": "train only", **p.__dict__},
        "splits": {},
    }

    for name, m in masks.items():
        args = (data["X"][m], data["label"][m], data["subject_id"][m],
                data["session_id"][m], data["clip_id"][m], data["frame_idx"][m])
        if name == "train" and m.any():
            Xa, ya, sa, ea, ca, fa, aug = _augment_train(*args, p)
            card["splits"][name] = _write(name, Xa, ya, sa, ea, ca, fa, aug)
        else:
            aug = np.zeros(int(m.sum()), dtype=bool)
            card["splits"][name] = _write(name, *args, aug)
        print(f"  {name}: {card['splits'][name]['rows']} rows")

    (OUT_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n-> {OUT_DIR}  (phase2_exit_met={valid_eval})")


if __name__ == "__main__":
    main()
