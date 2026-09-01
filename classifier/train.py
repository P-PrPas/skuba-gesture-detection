"""Phase 3/4: train a candidate classifier for the single classifier slot.

    python -m classifier.train --model rf                 # one model
    python -m classifier.train --model all                # every candidate
    python -m classifier.train --model catboost --features derived

Trains on data/dataset/train.npz, saves classifier/models/<model>.joblib
({clf, classes, clip, features}) + a meta json. `data/dataset/val.npz` is used
for early stopping / temperature scaling where the model supports it. Evaluate
with `python -m classifier.evaluate --all`.

Feature values are clipped to +-10 before fitting (normalisation blows up when
the hips leave the frame — ARCHITECTURE.md).
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

CANDIDATES = ["rf", "lgbm", "catboost", "hgb", "et", "svm", "logreg", "mlp"]

# classes whose train rows are augmented s01/s02 only -> the model should need
# strong evidence before predicting one. sit + laying moved out when the COCO
# posture mine landed; squat's COCO labels overlapped sit so it stays aug-only.
AUG_ONLY = {"squat", "glico_pose"}
WEIGHT_MULT = {c: 0.35 for c in AUG_ONLY} | {"idle": 0.6}
SVM_SUBSAMPLE = 12000          # SVC is ~O(n^2); cap the train set (screening)


def load_split(name: str, feat_mode: str):
    f = DS / f"{name}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    X = np.clip(d["X"].astype(np.float32), -CLIP, CLIP)
    return to_features(X, feat_mode), d["y"].astype(int)


def _weights(y):
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(y)
    base = compute_class_weight("balanced", classes=present, y=y)
    return {int(k): float(w) * WEIGHT_MULT.get(CLASSES[int(k)], 1.0)
            for k, w in zip(present, base)}


# --------------------------------------------------------------------------- #
def build_rf(y):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight=_weights(y), n_jobs=-1, random_state=0)


def build_et(y):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight=_weights(y), n_jobs=-1, random_state=0)


def build_lgbm(y):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective="multiclass", num_class=len(CLASSES),
                          n_estimators=400, learning_rate=0.05, num_leaves=63,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, class_weight=_weights(y),
                          n_jobs=-1, random_state=0, verbose=-1)


def build_catboost(y):
    from catboost import CatBoostClassifier
    # depth 6 (64 leaves/tree) — a screening config; the leading model gets tuned
    return CatBoostClassifier(loss_function="MultiClass", iterations=400,
                              learning_rate=0.06, depth=6,
                              class_weights=_weights(y), random_seed=0,
                              thread_count=-1, verbose=False, allow_writing_files=False)


def build_hgb(y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          class_weight=_weights(y),
                                          early_stopping=True, random_state=0)


def build_logreg(y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight=_weights(y), max_iter=2000,
                                            C=1.0, n_jobs=-1, random_state=0))


def build_svm(y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=4.0, gamma="scale", probability=True,
                             class_weight=_weights(y), random_state=0))


def build_mlp(y):
    from classifier.mlp import TorchMLP
    return TorchMLP(n_classes=len(CLASSES), class_weight=_weights(y), seed=0)


BUILDERS = {"rf": build_rf, "et": build_et, "lgbm": build_lgbm,
            "catboost": build_catboost, "hgb": build_hgb, "logreg": build_logreg,
            "svm": build_svm, "mlp": build_mlp}


# --------------------------------------------------------------------------- #
def train_one(model: str, feat_mode: str):
    MODELS.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_split("train", feat_mode)
    val = load_split("val", feat_mode)

    if model == "svm" and len(Xtr) > SVM_SUBSAMPLE:
        from sklearn.model_selection import train_test_split
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=SVM_SUBSAMPLE,
                                          stratify=ytr, random_state=0)
        sub = f" (subsampled to {SVM_SUBSAMPLE})"
    else:
        sub = ""
    print(f"[{model}] train {Xtr.shape}{sub}  features={feat_mode}")

    clf = BUILDERS[model](ytr)
    t0 = time.time()
    if model == "mlp" and val is not None:
        clf.fit(Xtr, ytr, val[0], val[1])
    else:
        clf.fit(Xtr, ytr)
    dt = time.time() - t0

    out = MODELS / f"{model}.joblib"
    joblib.dump({"clf": clf, "classes": CLASSES, "clip": CLIP, "features": feat_mode}, out)
    meta = {"model": model, "features": feat_mode, "train_rows": int(len(Xtr)),
            "train_dim": int(Xtr.shape[1]), "fit_seconds": round(dt, 1),
            "feature_clip": CLIP, "subsampled": bool(sub),
            "temperature": getattr(clf, "temperature", None)}
    (MODELS / f"{model}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[{model}] fit {dt:.1f}s -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=CANDIDATES + ["all"])
    ap.add_argument("--features", default="both",
                    choices=["raw", "derived", "both", "body_raw_hands_derived"])
    args = ap.parse_args()
    for m in (CANDIDATES if args.model == "all" else [args.model]):
        try:
            train_one(m, args.features)
        except Exception as e:  # noqa: BLE001
            print(f"[{m}] FAILED: {e}")
    print("\nnext: python -m classifier.evaluate --all")


if __name__ == "__main__":
    main()
