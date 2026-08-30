"""Phase 2 step 2: per-clip feature files -> versioned dataset + eval folds.

    python -m pipeline.build_dataset

Writes to data/dataset/:
  train.npz / val.npz / test.npz      subject-wise split (CLAUDE.md #5) when >=3
                                      subjects exist; otherwise all -> train.
  xperson_train.npz / xperson_test.npz   present when >=2 subjects overlap on a
                                      class: hold the "extra" subject out entirely.
                                      The only real cross-person number we can
                                      get from this dataset.
  dataset_card.json                   schema, backbone, aug params, per-class
                                      counts, split method, phase-2 status.

TRAIN splits are augmented (mirror+relabel, limb-length jitter, rotation,
dropout, coord noise - features/augment.py). Test/val never are.
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
FEAT_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "data" / "dataset"
SEED = 20260829
VAL_FRAC, TEST_FRAC = 0.15, 0.15
FIELDS = ("X", "label", "subject_id", "session_id", "clip_id", "frame_idx")


def _load_all() -> dict:
    files = sorted(FEAT_DIR.glob("*.npz"))
    if not files:
        raise SystemExit("no data/features/*.npz - run pipeline.extract_features first")
    parts = {k: [] for k in FIELDS}
    for p in files:
        d = np.load(p, allow_pickle=False)
        for k in parts:
            parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _subset(data: dict, mask: np.ndarray) -> tuple:
    return tuple(data[k][mask] for k in FIELDS)


def _split_subjects(subjects: np.ndarray) -> tuple[set, set, set]:
    uniq = sorted(set(subjects.tolist()))
    if len(uniq) < 3:
        return set(uniq), set(), set()
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_test = max(1, round(len(uniq) * TEST_FRAC))
    n_val = max(1, round(len(uniq) * VAL_FRAC))
    return set(uniq[n_val + n_test:]), set(uniq[:n_val]), set(uniq[n_val:n_val + n_test])


def _xperson_holdout(subjects: np.ndarray, labels: np.ndarray):
    """Pick the subject to hold out for the cross-person fold: the one with the
    fewest classes. Returns (subject, evaluable_classes) where evaluable_classes
    are the held subject's classes that OTHER subjects also cover (so training
    still sees them) - only those get a real cross-person score. None if no
    subject overlaps another on any class."""
    by_subj = defaultdict(set)
    for s, l in zip(subjects.tolist(), labels.tolist()):
        by_subj[s].add(l)
    if len(by_subj) < 2:
        return None
    for s, classes in sorted(by_subj.items(), key=lambda kv: len(kv[1])):
        others = set().union(*(c for k, c in by_subj.items() if k != s))
        evaluable = sorted(classes & others)
        if evaluable:
            return s, evaluable
    return None


def _augment(args: tuple, p: AugParams) -> tuple:
    X, y_str, subj, sess, clip, frame = args
    rng = np.random.default_rng(SEED + 1)
    out = {k: [] for k in (*FIELDS, "is_augmented")}
    for i in range(len(X)):
        for k, val in zip(FIELDS, (X[i], y_str[i], subj[i], sess[i], clip[i], frame[i])):
            out[k].append(val)
        out["is_augmented"].append(False)
        for _ in range(p.n_per_sample):
            v, lab = augment_once(X[i], str(y_str[i]), p, rng)
            for k, val in zip(FIELDS, (v, lab, subj[i], sess[i], clip[i], frame[i])):
                out[k].append(val)
            out["is_augmented"].append(True)
    return (np.stack(out["X"]).astype(np.float32), np.array(out["label"]),
            np.array(out["subject_id"]), np.array(out["session_id"]),
            np.array(out["clip_id"]), np.array(out["frame_idx"], dtype=np.int32),
            np.array(out["is_augmented"]))


def _write(name: str, X, y_str, subj, sess, clip, frame, aug) -> dict:
    y = np.array([class_index(str(l)) for l in y_str], dtype=np.int64) if len(y_str) else np.zeros(0, np.int64)
    np.savez_compressed(
        OUT_DIR / f"{name}.npz",
        X=X.astype(np.float32), y=y, label=y_str,
        subject_id=subj, session_id=sess, clip_id=clip, frame_idx=frame, is_augmented=aug,
    )
    counts = {c: int((y == class_index(c)).sum()) for c in CLASSES if (y == class_index(c)).any()}
    return {"rows": int(len(X)), "subjects": sorted(set(subj.tolist())),
            "augmented": bool(np.any(aug)), "class_counts": counts}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.npz"):
        stale.unlink()
    data = _load_all()
    assert data["X"].shape[1] == FEATURE_DIM, data["X"].shape
    subs = sorted(set(data["subject_id"].tolist()))
    p = AugParams()

    tr, va, te = _split_subjects(data["subject_id"])
    valid_eval = bool(va and te)
    print(f"subjects {subs} -> train={sorted(tr)} val={sorted(va)} test={sorted(te)}")

    card = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pose_backbone": json.loads((FEAT_DIR / "_extract_summary.json").read_text())["pose_backbone"],
        "feature_dim": FEATURE_DIM,
        "feature_layout": "body[0:66] lh[66:108] lh_present[108] rh[109:151] rh_present[151]",
        "classes": CLASSES,
        "n_subjects": len(subs),
        "split_method": "by_subject" if valid_eval else "all_to_train (<3 subjects)",
        "phase2_exit_met": valid_eval,
        "augmentation": {"applied_to": "train splits only", **p.__dict__},
        "splits": {},
    }

    # main split
    for name, keep in (("train", tr), ("val", va), ("test", te)):
        m = np.isin(data["subject_id"], list(keep))
        if not m.any():
            card["splits"][name] = {"rows": 0, "subjects": [], "augmented": False, "class_counts": {}}
            print(f"  {name}: 0 rows")
            continue
        args = _subset(data, m)
        parts = _augment(args, p) if name == "train" else (*args, np.zeros(int(m.sum()), bool))
        card["splits"][name] = _write(name, *parts)
        print(f"  {name}: {card['splits'][name]['rows']} rows")

    # cross-person diagnostic fold
    picked = None if valid_eval else _xperson_holdout(data["subject_id"], data["label"])
    if picked is not None:
        hold, evaluable = picked
        te_mask = data["subject_id"] == hold
        held_all = sorted(set(data["label"][te_mask].tolist()))
        card["splits"]["xperson_test"] = _write(
            "xperson_test", *_subset(data, te_mask), np.zeros(int(te_mask.sum()), bool))
        card["splits"]["xperson_train"] = _write(
            "xperson_train", *_augment(_subset(data, ~te_mask), p))
        card["xperson_fold"] = {
            "held_out_subject": hold,
            "test_classes": held_all,
            "evaluable_classes": evaluable,
            "holdout_only_classes": sorted(set(held_all) - set(evaluable)),
            "note": "Train on everyone else, test on the held-out subject. Score ONLY "
                    "evaluable_classes as a real cross-person number - holdout_only_classes "
                    "are absent from xperson_train so the model cannot predict them.",
        }
        print(f"  xperson: hold out {hold}; real cross-person score for {evaluable}; "
              f"train-only: {card['xperson_fold']['holdout_only_classes']}")

    if not valid_eval:
        card["phase2_status"] = (
            f"{len(subs)} subjects (need >=3 for a full subject-wise split). Mitigation: "
            "limb-length jitter augmentation to widen the body-proportion distribution; "
            "the xperson fold above for a real cross-person number on its classes; "
            "frozen keypoint backbone + shoulder-hip normalization limit subject leakage. "
            "Phase 6 field testing is the real generalization gate."
        )

    (OUT_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n-> {OUT_DIR}  (phase2_exit_met={valid_eval})")


if __name__ == "__main__":
    main()
