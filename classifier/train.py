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

from features.schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"
CLIP = 10.0


def load_split(name: str):
    d = np.load(DS / f"{name}.npz", allow_pickle=True)
    X = np.clip(d["X"].astype(np.float32), -CLIP, CLIP)
    return X, d["y"].astype(int), d


def build_lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES),
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, class_weight="balanced",
        n_jobs=-1, random_state=0,
    )


def build_rf():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, _ = load_split("train")
    print(f"train: {Xtr.shape}  classes present: {sorted(set(ytr.tolist()))}")

    clf = build_lgbm() if args.model == "lgbm" else build_rf()
    t0 = time.time()
    clf.fit(Xtr, ytr)
    dt = time.time() - t0

    out = MODELS / f"{args.model}.joblib"
    joblib.dump({"clf": clf, "classes": CLASSES, "clip": CLIP}, out)
    meta = {
        "model": args.model, "train_rows": int(len(Xtr)), "fit_seconds": round(dt, 1),
        "feature_clip": CLIP,
        "params": clf.get_params(),
    }
    (MODELS / f"{args.model}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"fit in {dt:.1f}s -> {out}")
    print("next: python -m classifier.evaluate --model", args.model)


if __name__ == "__main__":
    main()
