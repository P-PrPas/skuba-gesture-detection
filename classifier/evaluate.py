"""Evaluation harness — model-agnostic, reused for every classifier candidate.

    python -m classifier.evaluate --model rf
    python -m classifier.evaluate --all          # every trained model + comparison

A "model" is a bundle dict {clf, classes, clip, features} saved by
`classifier.train`; `clf` only needs `.predict_proba(X)` and `.classes_`. Train a
new candidate (MLP, a different tree lib, ...) into that shape and this harness
scores it identically — same metrics, same figures, same JSON schema.

Per model it writes:
  classifier/models/<m>.eval.json           full metrics (see `_evaluate`)
  results/phase3/fig/<m>_confusion.png      row-normalised confusion heatmap
  results/phase3/fig/<m>_perclass.png       precision / recall / F1 bars
  results/phase3/fig/<m>_calibration.png    reliability diagram + conf histogram
With --all also:
  results/phase3/fig/compare_f1.png         per-class F1, all models side by side
  results/phase3/fig/compare_overall.png    headline metrics, all models
  results/phase3/model_comparison.json

Metrics meaning by eval_type (from the dataset card) — only `cross_domain` and
`held_out_external` are real generalisation numbers; `aug_only_leak` train and
test overlap (except `sit`@s02 historically, now `squat`@nothing).
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
FIG = ROOT / "results" / "phase3" / "fig"

REAL_EVAL = ("cross_domain", "held_out_external")
_N_BINS = 10


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def _prf(y_true, y_pred, k):
    tp = int(((y_true == k) & (y_pred == k)).sum())
    fp = int(((y_true != k) & (y_pred == k)).sum())
    fn = int(((y_true == k) & (y_pred != k)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fn


def _calibration(conf, correct):
    """10-bin reliability diagram + expected calibration error."""
    edges = np.linspace(0.0, 1.0, _N_BINS + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if not n:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0,
                         "acc": None, "conf": None})
            continue
        acc = float(correct[m].mean())
        mc = float(conf[m].mean())
        ece += n / len(conf) * abs(acc - mc)
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n,
                     "acc": round(acc, 3), "conf": round(mc, 3)})
    return bins, round(ece, 3)


def _evaluate(name: str, split: str = "test") -> dict:
    bundle = joblib.load(MODELS / f"{name}.joblib")
    clf, clip = bundle["clf"], bundle["clip"]
    feat_mode = bundle.get("features", "raw")
    card = json.loads((DS / "dataset_card.json").read_text())
    per_class_eval = {c: card["per_class"].get(c, {}).get("eval_type", "-")
                      for c in CLASSES}

    d = np.load(DS / f"{split}.npz", allow_pickle=True)
    Xall = to_features(np.clip(d["X"].astype(np.float32), -clip, clip), feat_mode)
    yall = d["y"].astype(int)
    subj = d["subject_id"]

    model_classes = np.asarray(clf.classes_)
    modelled = set(model_classes.tolist())
    keep = np.isin(yall, list(modelled))
    X, y, subj = Xall[keep], yall[keep], subj[keep]

    proba = clf.predict_proba(X)
    pred = model_classes[proba.argmax(1)]
    conf = proba.max(1)
    correct = pred == y

    # per-class
    rows = {}
    for k in sorted(modelled):
        c = CLASSES[k]
        m = y == k
        if not m.any():
            continue
        prec, rec, f1, n = _prf(y, pred, k)
        cm = conf[m]
        cc = conf[m & correct]
        cw = conf[m & ~correct]
        rows[c] = {
            "eval_type": per_class_eval[c], "n": n,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "mean_conf": round(float(cm.mean()), 3),
            "mean_conf_correct": round(float(cc.mean()), 3) if len(cc) else None,
            "mean_conf_wrong": round(float(cw.mean()), 3) if len(cw) else None,
        }

    real = [c for c, r in rows.items() if r["eval_type"] in REAL_EVAL]
    xdom = [c for c, r in rows.items() if r["eval_type"] == "cross_domain"]
    f1s = np.array([r["f1"] for r in rows.values()])
    macro = float(f1s.mean())

    # confusion (row-normalised + counts), ordered by CLASSES
    order = [k for k in range(len(CLASSES)) if k in modelled and (y == k).any()]
    labels = [CLASSES[k] for k in order]
    cm_counts = np.zeros((len(order), len(order)), int)
    idx = {k: i for i, k in enumerate(order)}
    for t, p in zip(y, pred):
        if t in idx and p in idx:
            cm_counts[idx[t], idx[p]] += 1
    cm_norm = cm_counts / cm_counts.sum(1, keepdims=True).clip(min=1)

    # cross-person: any class present for >1 subject
    xperson = {}
    for k in sorted(modelled):
        ss = sorted(set(subj[y == k].tolist()))
        if len(ss) > 1:
            xperson[CLASSES[k]] = {s: round(float((pred[(y == k) & (subj == s)] == k).mean()), 3)
                                   for s in ss}

    bins, ece = _calibration(conf, correct)
    mj = MODELS / f"{name}.meta.json"
    meta = json.loads(mj.read_text()) if mj.exists() else {}
    jb = MODELS / f"{name}.joblib"

    top_conf = sorted(((int(cm_counts[i, j]), labels[i], labels[j])
                       for i in range(len(labels)) for j in range(len(labels)) if i != j),
                      reverse=True)[:12]

    return {
        "model": name,
        "split": split,
        "features": feat_mode,
        "n_test_rows": int(len(yall)),
        "n_scored_rows": int(len(y)),
        "n_modelled_classes": len(rows),
        "macro_f1": round(macro, 3),
        "macro_f1_real_eval": round(float(np.mean([rows[c]["f1"] for c in real])), 3) if real else None,
        "macro_f1_cross_domain": round(float(np.mean([rows[c]["f1"] for c in xdom])), 3) if xdom else None,
        "micro_f1_accuracy": round(float(correct.mean()), 3),
        "balanced_accuracy": round(float(np.mean([rows[c]["recall"] for c in rows])), 3),
        "mean_confidence": round(float(conf.mean()), 3),
        "mean_conf_correct": round(float(conf[correct].mean()), 3),
        "mean_conf_wrong": round(float(conf[~correct].mean()), 3) if (~correct).any() else None,
        "expected_calibration_error": ece,
        "fit_seconds": meta.get("fit_seconds"),
        "model_size_mb": round(jb.stat().st_size / 1e6, 1) if jb.exists() else None,
        "n_features": meta.get("train_dim"),
        "train_rows": meta.get("train_rows"),
        "per_class": rows,
        "cross_person": xperson,
        "calibration_bins": bins,
        "confusion_labels": labels,
        "confusion_counts": cm_counts.tolist(),
        "confusion_row_normalised": np.round(cm_norm, 3).tolist(),
        "top_confusions": [{"true": a, "pred": b, "count": n} for n, a, b in top_conf if n],
    }


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #
_ET_COLOR = {"cross_domain": "#1a7f37", "held_out_external": "#0969da",
             "aug_only_leak": "#bf8700", "-": "#8250df"}


def _confusion_png(ev: dict, path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ev["confusion_labels"]
    norm = np.array(ev["confusion_row_normalised"])
    counts = np.array(ev["confusion_counts"])
    ann = np.where(counts > 0, counts.astype(str), "")
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(labels), 0.9 + 0.55 * len(labels)))
    sns.heatmap(norm, annot=ann, fmt="", cmap="rocket_r", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "row fraction"},
                linewidths=0.5, linecolor="#ddd", ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{ev['model'].upper()} — confusion (row-normalised, counts shown)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _perclass_png(ev: dict, path: Path):
    import matplotlib.pyplot as plt

    cs = list(ev["per_class"])
    pr = [ev["per_class"][c]["precision"] for c in cs]
    rc = [ev["per_class"][c]["recall"] for c in cs]
    f1 = [ev["per_class"][c]["f1"] for c in cs]
    x = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(2.0 + 0.85 * len(cs), 4.6))
    ax.bar(x - 0.25, pr, 0.24, label="precision", color="#9ecae1")
    ax.bar(x, rc, 0.24, label="recall", color="#4292c6")
    ax.bar(x + 0.25, f1, 0.24, label="F1", color="#08519c")
    for xi, c in zip(x, cs):                       # eval-type strip under each group
        ax.axvspan(xi - 0.4, xi + 0.4, ymin=0, ymax=0.03,
                   color=_ET_COLOR.get(ev["per_class"][c]["eval_type"], "#888"))
    ax.axhline(0.6, ls="--", lw=0.9, color="#c00", label="0.6 floor")
    ax.set_xticks(x)
    ax.set_xticklabels(cs, rotation=45, ha="right")
    for lbl, c in zip(ax.get_xticklabels(), cs):
        lbl.set_color(_ET_COLOR.get(ev["per_class"][c]["eval_type"], "#333"))
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    ax.set_title(f"{ev['model'].upper()} — per-class  "
                 "(label colour = eval type: green cross-domain / blue held-out / amber leak)")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _calibration_png(ev: dict, path: Path):
    import matplotlib.pyplot as plt

    b = [x for x in ev["calibration_bins"] if x["n"]]
    mid = [(x["lo"] + x["hi"]) / 2 for x in b]
    acc = [x["acc"] for x in b]
    n = [x["n"] for x in b]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot([0, 1], [0, 1], ls="--", color="#999", label="perfect")
    ax1.plot(mid, acc, "o-", color="#08519c", label="observed")
    ax1.set_xlabel("mean confidence in bin"); ax1.set_ylabel("accuracy in bin")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_title(f"reliability (ECE={ev['expected_calibration_error']})")
    ax1.legend(loc="upper left")
    ax2.bar(mid, n, width=0.08, color="#6baed6")
    ax2.set_xlabel("confidence"); ax2.set_ylabel("# test frames"); ax2.set_xlim(0, 1)
    ax2.set_title(f"{ev['model'].upper()} — confidence histogram")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _compare_png(evs: list[dict], f1_path: Path, overall_path: Path):
    import matplotlib.pyplot as plt

    classes = list(evs[0]["per_class"])
    x = np.arange(len(classes))
    w = 0.8 / len(evs)
    fig, ax = plt.subplots(figsize=(2 + 0.8 * len(classes), 4.2))
    for i, ev in enumerate(evs):
        ax.bar(x + i * w - 0.4 + w / 2,
               [ev["per_class"].get(c, {}).get("f1", 0) for c in classes], w,
               label=ev["model"].upper())
    ax.axhline(0.6, ls="--", lw=0.8, color="#c00")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.legend()
    ax.set_title("per-class F1 — baseline models")
    fig.tight_layout(); fig.savefig(f1_path, dpi=130); plt.close(fig)

    keys = [("macro_f1", "macro-F1 (all)"), ("macro_f1_real_eval", "macro-F1 (real)"),
            ("micro_f1_accuracy", "accuracy"), ("balanced_accuracy", "balanced acc"),
            ("expected_calibration_error", "ECE (lower=better)")]
    xk = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, ev in enumerate(evs):
        ax.bar(xk + i * w - 0.4 + w / 2, [ev.get(k) or 0 for k, _ in keys], w,
               label=ev["model"].upper())
    ax.set_xticks(xk); ax.set_xticklabels([lbl for _, lbl in keys], rotation=20, ha="right")
    ax.set_ylabel("value"); ax.legend(); ax.set_title("headline metrics — baseline models")
    fig.tight_layout(); fig.savefig(overall_path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def _run_one(name: str, split: str = "test") -> dict:
    ev = _evaluate(name, split)
    tag = "" if split == "test" else f".{split}"
    (MODELS / f"{name}{tag}.eval.json").write_text(json.dumps(ev, indent=2))
    FIG.mkdir(parents=True, exist_ok=True)
    _confusion_png(ev, FIG / f"{name}{tag}_confusion.png")
    _perclass_png(ev, FIG / f"{name}{tag}_perclass.png")
    _calibration_png(ev, FIG / f"{name}{tag}_calibration.png")
    print(f"\n== {name} ==  macro-F1 {ev['macro_f1']}  real-eval {ev['macro_f1_real_eval']}  "
          f"acc {ev['micro_f1_accuracy']}  ECE {ev['expected_calibration_error']}  "
          f"{ev['model_size_mb']} MB  fit {ev['fit_seconds']}s")
    for c, r in ev["per_class"].items():
        print(f"  {c:16s} {r['eval_type']:18s} n={r['n']:4d}  "
              f"P {r['precision']:.2f}  R {r['recall']:.2f}  F1 {r['f1']:.2f}  "
              f"conf {r['mean_conf']:.2f}")
    for c, w in ev["cross_person"].items():
        print(f"  cross-person {c}: {w}")
    print(f"  -> {MODELS / f'{name}.eval.json'} + 3 figures in results/phase3/fig/")
    return ev


# canonical order for figures/tables (best-known first)
_ORDER = ["rf", "et", "lgbm", "catboost", "hgb", "svm", "logreg", "mlp", "gcn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="one model name")
    ap.add_argument("--all", action="store_true",
                    help="every trained classifier/models/*.joblib + comparison")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="which split to score (default test)")
    args = ap.parse_args()

    if args.all:
        found = {p.stem for p in MODELS.glob("*.joblib")}
        names = [n for n in _ORDER if n in found] + sorted(found - set(_ORDER))
    else:
        names = [args.model or "lgbm"]
    names = [n for n in names if (MODELS / f"{n}.joblib").exists()]
    if not names:
        raise SystemExit("no trained model — run classifier.train first")

    evs = [_run_one(n, args.split) for n in names]

    if len(evs) > 1:
        # >2 models is a Phase 4 comparison -> results/phase4/; the RF-vs-LGBM
        # Phase 3 comparison stays in results/phase3/.
        cdir = ROOT / "results" / ("phase3" if set(names) <= {"rf", "lgbm"} else "phase4")
        cfig = cdir / "fig"
        cfig.mkdir(parents=True, exist_ok=True)
        tag = "" if args.split == "test" else f".{args.split}"
        _compare_png(evs, cfig / f"compare_f1{tag}.png", cfig / f"compare_overall{tag}.png")
        cmp = {"generated_from": [e["model"] for e in evs], "split": args.split,
               "headline": {e["model"]: {k: e[k] for k in
                            ("macro_f1", "macro_f1_real_eval", "macro_f1_cross_domain",
                             "micro_f1_accuracy", "balanced_accuracy",
                             "expected_calibration_error", "mean_conf_correct",
                             "mean_conf_wrong", "model_size_mb", "fit_seconds")}
                            for e in evs},
               "per_class_f1": {c: {e["model"]: e["per_class"].get(c, {}).get("f1")
                                    for e in evs}
                                for c in evs[0]["per_class"]},
               "cross_person": {e["model"]: e["cross_person"] for e in evs}}
        out = cdir / f"model_comparison{tag}.json"
        out.write_text(json.dumps(cmp, indent=2))
        print(f"\n-> {out} + compare figures in {cfig}")


if __name__ == "__main__":
    main()
