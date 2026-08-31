"""Phase 3: evaluate the baseline on data/dataset/test.npz.

    python -m classifier.evaluate --model lgbm

Prints per-class precision/recall/F1, the confusion matrix, and a breakdown by
eval type (cross_domain vs held_out_external vs aug_only_leak) and by subject
for `sit`. Writes classifier/models/<model>.eval.json.

Reported numbers mean different things per class - see dataset_card.json
`per_class[...].eval_type`. Only cross_domain + `sit`@s02 are real
generalisation numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from features.derived import to_features
from features.schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"


def _prf(y_true, y_pred, k):
    tp = int(((y_true == k) & (y_pred == k)).sum())
    fp = int(((y_true != k) & (y_pred == k)).sum())
    fn = int(((y_true == k) & (y_pred != k)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    args = ap.parse_args()

    bundle = joblib.load(MODELS / f"{args.model}.joblib")
    clf, clip = bundle["clf"], bundle["clip"]
    feat_mode = bundle.get("features", "raw")
    card = json.loads((DS / "dataset_card.json").read_text())
    per_class = card["per_class"]

    d = np.load(DS / "test.npz", allow_pickle=True)
    Xall = to_features(np.clip(d["X"].astype(np.float32), -clip, clip), feat_mode)
    yall = d["y"].astype(int)
    subjall = d["subject_id"]
    model_classes = np.asarray(clf.classes_)          # subset of CLASSES indices
    modelled = set(model_classes.tolist())

    # pending-data classes have no label in this model — score only the frames
    # of classes the model actually knows (feeding it i_love_you frames and
    # penalising the result is not meaningful). Report where they land separately.
    keep = np.isin(yall, list(modelled))
    X, y, subj = Xall[keep], yall[keep], subjall[keep]
    proba = clf.predict_proba(X)
    pred = model_classes[proba.argmax(1)]
    conf = proba.max(1)

    pend_land = {}
    if (~keep).any():
        pp = model_classes[clf.predict_proba(Xall[~keep]).argmax(1)]
        for k in sorted(set(yall[~keep].tolist())):
            u, c = np.unique(pp[yall[~keep] == k], return_counts=True)
            pend_land[CLASSES[k]] = {CLASSES[int(a)]: int(b)
                                     for a, b in sorted(zip(u, c), key=lambda z: -z[1])}

    print(f"\n== {args.model} on test — {len(y)} rows over {len(modelled)} modelled classes ==\n")
    print(f"{'class':16s} {'eval':18s} {'n':>4s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'meanconf':>8s}")
    rows = {}
    for k, c in enumerate(CLASSES):
        if k not in modelled:
            continue
        m = y == k
        if not m.any():
            continue
        prec, rec, f1, n = _prf(y, pred, k)
        mc = float(conf[m].mean())
        et = per_class.get(c, {}).get("eval_type", "-")
        print(f"{c:16s} {et:18s} {n:4d} {prec:5.2f} {rec:5.2f} {f1:5.2f} {mc:8.2f}")
        rows[c] = {"eval_type": et, "n": n, "precision": round(prec, 3),
                   "recall": round(rec, 3), "f1": round(f1, 3), "mean_conf": round(mc, 3)}

    real = [c for c, r in rows.items() if r["eval_type"] in ("cross_domain", "held_out_external")]
    macro_real = float(np.mean([rows[c]["f1"] for c in real]))
    macro_modelled = float(np.mean([r["f1"] for r in rows.values()]))
    macro_all = macro_modelled
    print(f"\nmacro-F1  modelled ({len(rows)}): {macro_modelled:.3f}   "
          f"real-eval only ({len(real)}): {macro_real:.3f}")
    if pend_land:
        print("\npending-data classes (no label in this model) land as:")
        for c, w in pend_land.items():
            print(f"  {c:14s} -> {w}")

    # sit: s01 (leak) vs s02 (clean cross-person)
    if "sit" in rows:
        ksit = CLASSES.index("sit")
        for s in sorted(set(subj[y == ksit].tolist())):
            mm = (y == ksit) & (subj == s)
            acc = float((pred[mm] == ksit).mean())
            print(f"  sit @ {s}: recall {acc:.2f} ({mm.sum()} frames)"
                  + ("   <- clean cross-person" if s == "s02" else "   (leak)"))

    # confusion: which wrong class each true class most often becomes
    print("\ntop confusions (true -> predicted, count):")
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, pr in zip(y, pred):
        cm[t, pr] += 1
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j]:
                pairs.append((cm[i, j], CLASSES[i], CLASSES[j]))
    for cnt, a, b in sorted(pairs, reverse=True)[:12]:
        print(f"  {a:16s} -> {b:16s} {cnt}")

    out = {
        "model": args.model, "features": feat_mode,
        "macro_f1_modelled": round(macro_modelled, 3),
        "macro_f1_all": round(macro_all, 3),
        "macro_f1_real_eval": round(macro_real, 3),
        "pending_data_lands_as": pend_land, "per_class": rows,
        "confusion": {CLASSES[i]: {CLASSES[j]: int(cm[i, j]) for j in range(len(CLASSES)) if cm[i, j]}
                      for i in range(len(CLASSES)) if cm[i].any()},
    }
    (MODELS / f"{args.model}.eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {MODELS / f'{args.model}.eval.json'}")


if __name__ == "__main__":
    main()
