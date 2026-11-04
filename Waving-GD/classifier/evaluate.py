"""Evaluate on the user's two held-out clips (never used for training) vs the plain geometric rule.

    python -m classifier.evaluate --name v2_all_fpr05 [--name v2_raw_fpr05 ...]   # also writes outputs/runs/<name>/eval_test.txt

Needs: data/kpts/test/*.npz (pipeline.extract_features test) and data/test_segments.csv
(label 1/0, -1 = transition/ambiguous and ignored). The deployment rate is simulated by sub-sampling frames to
--fps (default 5) and applying the same TrackSmoother as the detector.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from classifier.model import Classifier
from features.keypoints import features_from_raw, rule_raised
from features.temporal import TrackSmoother


def segments():
    seg = {}
    for r in csv.DictReader(open(C.ROOT / "data" / "test_segments.csv")):
        seg.setdefault(r["clip"], []).append((float(r["t_start"]), float(r["t_end"]), int(r["label"])))
    return seg


def gt_at(segs, t):
    for a, b, l in segs:
        if a <= t < b:
            return l
    return -1


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / max(len(y), 1), P=pr, R=rc, F1=2 * pr * rc / max(pr + rc, 1e-9), FP=fp, FN=fn, n=len(y))


def timeline(t, y, p, width=60):
    """one char per time bucket: '#' raise & right, 'x' error, '.' not-raise & right, ' ' ignored"""
    out = []
    for lo in np.linspace(t[0], t[-1], width, endpoint=False):
        m = (t >= lo) & (t < lo + (t[-1] - t[0]) / width)
        if not m.any() or (y[m] < 0).all():
            out.append(" "); continue
        v = y[m] >= 0; yy, pp = y[m][v], p[m][v]
        out.append("x" if (yy != pp).mean() > 0.5 else ("#" if yy.mean() > 0.5 else "."))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", default=[]); ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=3, help="smoothing window in processed frames")
    a = ap.parse_args()
    import io
    buf = io.StringIO(); real = sys.stdout

    class T:
        def write(s, x): real.write(x); buf.write(x)
        def flush(s): real.flush()
    sys.stdout = T()
    seg = segments(); clips = sorted((C.KPTS / "test").glob("*.npz"))
    models = {n: Classifier(C.model_path(n)) for n in a.name}
    res = {k: {"y": [], "p": []} for k in ["rule", "rule+smooth", *models, *[n + "+smooth" for n in models]]}
    for c in clips:
        d = np.load(c); fps = float(d["fps"]); step = max(int(round(fps / a.fps)), 1)
        idx = np.arange(0, len(d["kpt"]), step); t = d["t"][idx]
        y = np.array([gt_at(seg[c.stem], x) for x in t])
        F = np.stack([features_from_raw(d["kpt"][i], d["box"][i]) for i in idx])
        sm = {k: TrackSmoother(a.window) for k in res}
        preds = {"rule": np.array([rule_raised(d["kpt"][i]) for i in idx], int)}
        preds["rule+smooth"] = np.array([sm["rule+smooth"].update(0, v) >= 0.5 for v in preds["rule"]], int)
        for n, m in models.items():
            pr = m.predict(F); preds[n] = (pr >= m.thr).astype(int)
            preds[n + "+smooth"] = np.array([sm[n + "+smooth"].update(0, v) >= m.thr for v in pr], int)
        print(f"\n{c.stem}  ({len(idx)} frames @ ~{a.fps:g} FPS, {int((y >= 0).sum())} labelled)  '#'=raise ok  '.'=not-raise ok  'x'=error")
        print(f"{'ground truth':16s} |" + "".join("#" if v == 1 else ("." if v == 0 else " ") for v in np.interp(np.linspace(t[0], t[-1], 60, endpoint=False), t, y).round()) + "|")
        for k, p in preds.items():
            print(f"{k:16s} |{timeline(t, y, p)}|")
            v = y >= 0; res[k]["y"] += list(y[v]); res[k]["p"] += list(p[v])
    print("\n== overall on labelled frames (both clips)")
    print(f"{'method':18s} {'acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'FP':>4s} {'FN':>4s} {'n':>5s}")
    for k, v in res.items():
        m = metrics(v["y"], v["p"])
        print(f"{k:18s} {m['acc']:6.1%} {m['P']:6.1%} {m['R']:6.1%} {m['F1']:6.3f} {m['FP']:4d} {m['FN']:4d} {m['n']:5d}")
    sys.stdout = real
    for n in a.name:
        (C.RUNS / n / "eval_test.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
