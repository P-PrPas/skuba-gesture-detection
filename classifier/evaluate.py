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
    card = json.loads((DS / "dataset_card.json").read_text())
    per_class = card["per_class"]

    d = np.load(DS / "test.npz", allow_pickle=True)
    X = np.clip(d["X"].astype(np.float32), -clip, clip)
    y = d["y"].astype(int)
    subj = d["subject_id"]
    proba = clf.predict_proba(X)
    pred = proba.argmax(1)
    conf = proba.max(1)

    print(f"\n== {args.model} on test ({len(y)} rows) ==\n")
    print(f"{'class':16s} {'eval':18s} {'n':>4s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'meanconf':>8s}")
    rows = {}
    for k, c in enumerate(CLASSES):
        m = y == k
        if not m.any():
            continue
        prec, rec, f1, n = _prf(y, pred, k)
        mc = float(conf[m].mean())
        et = per_class.get(c, {}).get("eval_type", "-")
        print(f"{c:16s} {et:18s} {n:4d} {prec:5.2f} {rec:5.2f} {f1:5.2f} {mc:8.2f}")
        rows[c] = {"eval_type": et, "n": n, "precision": round(prec, 3),
                   "recall": round(rec, 3), "f1": round(f1, 3), "mean_conf": round(mc, 3)}

    # headline: macro-F1 over the classes that are a real generalisation test
    real = [c for c, r in rows.items() if r["eval_type"] in ("cross_domain", "held_out_external")]
    macro_real = float(np.mean([rows[c]["f1"] for c in real]))
    macro_all = float(np.mean([r["f1"] for r in rows.values()]))
    print(f"\nmacro-F1  all classes: {macro_all:.3f}   real-eval only ({len(real)}): {macro_real:.3f}")

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
        "model": args.model, "macro_f1_all": round(macro_all, 3),
        "macro_f1_real_eval": round(macro_real, 3), "per_class": rows,
        "confusion": {CLASSES[i]: {CLASSES[j]: int(cm[i, j]) for j in range(len(CLASSES)) if cm[i, j]}
                      for i in range(len(CLASSES)) if cm[i].any()},
    }
    (MODELS / f"{args.model}.eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {MODELS / f'{args.model}.eval.json'}")


if __name__ == "__main__":
    main()
