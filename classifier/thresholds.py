"""Phase 4 Track E: confidence-threshold analysis for the idle/unknown fallback.

    python -m classifier.thresholds --model mlp

Rule under test (Phase 5 fallback): for a frame whose argmax class is c, if the
max probability < T[c] the frame is emitted as `idle` instead of c.

FINDING (2026-09-01, locked MLP): per-class thresholds fit on `val` do NOT
transfer to `test`. `val` is the same webcam domain as train, so its
confidences run ~0.2 higher than the real cross-domain `test` frames; a
threshold that keeps 85 % of val true-positives rejects a large slice of
*correct* test predictions (mini_heart collapses to F1 0, laying 0.89->0.68).
And on `test` the idle errors are real gestures misread as idle — a
reject-to-idle gate makes idle precision worse, not better, at every threshold.

Conclusion: no static reject threshold above a garbage floor (~0.20) helps on
the honest split. The frame-level idle<->gesture flicker is Phase 5's temporal
majority vote to fix, and the idle over-trigger is a data/feature gap for the
Phase 6 field set. This script stays as the harness to re-tune once real
field confidences exist.

Writes classifier/models/<model>.thresholds.json:
  global_floor            the only active rule — drop sub-0.20 frames to idle
  reference_val_pXX       per-class val percentile thresholds, NOT active
                          (kept for the record + Phase 5 re-tuning)
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
IDLE = CLASSES.index("idle")
GLOBAL_FLOOR = 0.20         # the only rule that survives the val->test transfer check


def _load(split, feat_mode, clip):
    d = np.load(DS / f"{split}.npz", allow_pickle=True)
    X = to_features(np.clip(d["X"].astype(np.float32), -clip, clip), feat_mode)
    return X, d["y"].astype(int)


def _f1(y, pred, k):
    tp = int(((y == k) & (pred == k)).sum())
    fp = int(((y != k) & (pred == k)).sum())
    fn = int(((y == k) & (pred != k)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def _macro(y, pred):
    return float(np.mean([_f1(y, pred, k)[0] for k in np.unique(y)]))


def _apply(pred, conf, thr_vec):
    g = pred.copy()
    g[(conf < thr_vec) & (pred != IDLE)] = IDLE
    return g


def demo():
    pred = np.array([1, 2, IDLE, 3])
    conf = np.array([0.9, 0.1, 0.05, 0.5])
    g = _apply(pred, conf, np.full(4, 0.2))
    assert g.tolist() == [1, IDLE, IDLE, 3], g          # only the 0.1 non-idle flips
    y = np.array([1, 1, 2, 2])
    assert _f1(y, np.array([1, 1, 2, 2]), 1)[0] == 1.0
    assert abs(_macro(y, np.array([1, 1, 1, 2])) - np.mean([0.8, 2/3])) < 1e-9
    print("thresholds.demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--keep", type=float, default=0.85,
                    help="per-class: keep this fraction of val TPs (reference only)")
    args = ap.parse_args()
    if args.demo:
        return demo()

    b = joblib.load(MODELS / f"{args.model}.joblib")
    clf, feat_mode, clip = b["clf"], b.get("features", "raw"), b["clip"]
    classes = np.asarray(clf.classes_)

    scored = {}
    for sp in ("val", "test"):
        X, y = _load(sp, feat_mode, clip)
        proba = clf.predict_proba(X)
        keep = np.isin(y, classes)
        y = y[keep]
        pred = classes[proba[keep].argmax(1)]
        conf = proba[keep].max(1)
        scored[sp] = (y, pred, conf)

    # per-class reference thresholds from val
    yv, pv, cv = scored["val"]
    correct = pv == yv
    ref = {}
    for k in classes:
        m = (pv == k) & correct
        ref[CLASSES[int(k)]] = (round(float(np.quantile(cv[m], 1.0 - args.keep)), 3)
                                if m.sum() >= 10 else None)

    # transfer check: apply the val per-class thresholds to both splits
    print(f"model={args.model}  features={feat_mode}")
    print(f"\nper-class val-p{int((1-args.keep)*100)} thresholds (reference, NOT active):")
    for c, t in ref.items():
        if t is not None:
            print(f"  {c:16s} {t}")

    for sp in ("val", "test"):
        y, pred, conf = scored[sp]
        thr = np.array([ref.get(CLASSES[int(c)]) or 0.0 for c in pred])
        g = _apply(pred, conf, thr)
        fl = _apply(pred, conf, np.full(len(pred), GLOBAL_FLOOR))
        ip = _f1(y, pred, IDLE)
        gp = _f1(y, g, IDLE)
        fp_ = _f1(y, fl, IDLE)
        print(f"\n[{sp}]  n={len(y)}")
        print(f"  raw           macroF1 {_macro(y, pred):.3f}  acc {(pred == y).mean():.3f}  "
              f"idle P/R {ip[1]:.2f}/{ip[2]:.2f}")
        print(f"  val-per-class macroF1 {_macro(y, g):.3f}  acc {(g == y).mean():.3f}  "
              f"idle P/R {gp[1]:.2f}/{gp[2]:.2f}  ({(g != pred).sum()} ->idle)")
        print(f"  global {GLOBAL_FLOOR}   macroF1 {_macro(y, fl):.3f}  acc {(fl == y).mean():.3f}  "
              f"idle P/R {fp_[1]:.2f}/{fp_[2]:.2f}  ({(fl != pred).sum()} ->idle)")

    out = MODELS / f"{args.model}.thresholds.json"
    out.write_text(json.dumps({
        "model": args.model, "features": feat_mode,
        "active_rule": f"argmax=c; if maxproba < {GLOBAL_FLOOR} emit idle",
        "global_floor": GLOBAL_FLOOR,
        "reference_val_percentile": {"keep": args.keep, "thresholds": ref,
                                     "active": False,
                                     "note": "val confidences run ~0.2 high vs "
                                     "cross-domain test; do not activate without "
                                     "real field-set calibration (Phase 6)"},
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
