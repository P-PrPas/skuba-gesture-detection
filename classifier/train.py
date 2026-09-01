"""Phase 3: baseline classifier on the fused 152-d features.

    python -m classifier.train                 # LightGBM (default)
    python -m classifier.train --model rf      # RandomForest

Trains on data/dataset/train.npz, saves classifier/models/<model>.joblib +
a small meta json. Evaluation is `classifier.evaluate`.

Feature values are clipped to +-10 before fitting — normalisation blows up on a
handful of frames where the hips are out of the camera frame (see
ARCHITECTURE.md); trees don't care but it keeps the MLP option honest later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from features.derived import to_features
from features.schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"
CLIP = 10.0


def load_split(name: str, feat_mode: str):
    d = np.load(DS / f"{name}.npz", allow_pickle=True)
    X = np.clip(d["X"].astype(np.float32), -CLIP, CLIP)
    X = to_features(X, feat_mode)
    return X, d["y"].astype(int), d


# Per-class weight multipliers on top of 'balanced'.
#   aug-only: one person's augmented frames, unreliable -> the model should need
#            strong evidence before predicting one.
#   idle: huge, and gesture frames near a class boundary fall into it (idle
#         recall is already 1.0, so trading a little of it for gesture recall
#         and idle precision is a good deal).
# classes whose train rows are augmented s01/s02 only -> the model should need
# strong evidence before predicting one. Drop a class from here once it gains a
# real external source (sit/squat/laying: when the COCO posture mine lands).
AUG_ONLY = {"sit", "laying", "squat", "glico_pose"}
WEIGHT_MULT = {c: 0.35 for c in AUG_ONLY} | {"idle": 0.6}


def _weights(y):
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(y)
    base = compute_class_weight("balanced", classes=present, y=y)
    return {int(k): float(w) * WEIGHT_MULT.get(CLASSES[int(k)], 1.0)
            for k, w in zip(present, base)}


def build_lgbm(y):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES),
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, class_weight=_weights(y),
        n_jobs=-1, random_state=0,
    )


def build_rf(y):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight=_weights(y), n_jobs=-1, random_state=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    ap.add_argument("--features", default="both",
                    choices=["raw", "derived", "both", "body_raw_hands_derived"])
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, _ = load_split("train", args.features)
    print(f"train: {Xtr.shape}  features={args.features}  "
          f"classes present: {sorted(set(ytr.tolist()))}")

    clf = build_lgbm(ytr) if args.model == "lgbm" else build_rf(ytr)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    dt = time.time() - t0

    out = MODELS / f"{args.model}.joblib"
    joblib.dump({"clf": clf, "classes": CLASSES, "clip": CLIP, "features": args.features}, out)
    meta = {
        "model": args.model, "features": args.features,
        "train_rows": int(len(Xtr)), "train_dim": int(Xtr.shape[1]),
        "fit_seconds": round(dt, 1), "feature_clip": CLIP,
        "params": clf.get_params(),
    }
    (MODELS / f"{args.model}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"fit in {dt:.1f}s -> {out}")
    print("next: python -m classifier.evaluate --model", args.model)


if __name__ == "__main__":
    main()
