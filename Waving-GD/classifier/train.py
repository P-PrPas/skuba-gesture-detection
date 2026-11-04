"""Train the MLP head on cached YOLO-pose features.

    python -m classifier.train --name v2_all_fpr05 --max-hn-fpr 0.05              # raw + derived distances/angles
    python -m classifier.train --name v2_raw_fpr05 --max-hn-fpr 0.05 --features raw   # ablation: keypoints only
Each --name is a version: outputs/runs/<name>/{model.pt, model.json, train.log} (never overwritten silently).

Augmentation: horizontal mirror (label unchanged: either arm = raise hand), gaussian noise.
Sampling weights: 1/sqrt(frames in the (source, role, label) group) so no single source dominates.
Threshold: best-F1 on val subject to false-positive rate on hard-negative/neutral frames <= --max-hn-fpr
(the cap is a ceiling, not a target); or with --min-recall R: highest threshold reaching val recall >= R.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from classifier.model import MLP, columns, save


class _Tee:
    def __init__(self, f): self.f, self.o = f, sys.__stdout__
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def tee_to(path):
    sys.stdout = _Tee(open(path, "w"))


def load(split):
    d = np.load(C.DATASET / f"{split}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def group_weights(d):
    key = np.char.add(np.char.add(d["source"], d["role"]), d["y"].astype(str))
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / np.sqrt(cnt[inv])
    pos = d["y"] == 1
    w[pos] *= w[~pos].sum() / max(w[pos].sum(), 1e-9)   # total positive mass == total negative mass
    return (w / w.mean()).astype(np.float32)


def pick_threshold(p, y, neg_mask, max_fpr, min_recall=0.0):
    """Default: best F1 with hard-neg FPR <= max_fpr.
    With min_recall > 0: the HIGHEST threshold whose val recall >= min_recall (lowest FPR for that recall);
    falls back to the cap-only rule (and warns) if the FPR cap makes that impossible."""
    if min_recall > 0:
        for t in np.linspace(0.95, 0.05, 91):
            pred = p >= t
            rec = pred[y == 1].mean(); fpr = pred[neg_mask].mean() if neg_mask.any() else 0.0
            if rec >= min_recall:
                if fpr <= max_fpr:
                    tp = (pred & (y == 1)).sum(); prec = tp / max(pred.sum(), 1)
                    return (float(t), float(2 * prec * rec / max(prec + rec, 1e-9)), float(fpr), float(prec), float(rec))
                print(f"WARNING: recall>={min_recall:.0%} needs hard-neg FPR {fpr:.1%} > cap {max_fpr:.1%}; falling back to best-F1 rule")
                break
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        fpr = pred[neg_mask].mean() if neg_mask.any() else 0.0
        tp = (pred & (y == 1)).sum(); prec = tp / max(pred.sum(), 1); rec = tp / max((y == 1).sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if fpr <= max_fpr and (best is None or f1 > best[1]):
            best = (float(t), float(f1), float(fpr), float(prec), float(rec))
    return best or (0.95, 0.0, float(neg_mask.mean()), 0.0, 0.0)


def report(p, d, thr, title):
    y = d["y"]; pred = p >= thr
    print(f"\n== {title}  (thr={thr:.2f})")
    print(f"{'source/role':24s} {'n':>7s} {'FP rate' :>8s} {'recall':>8s}")
    for so in np.unique(d["source"]):
        for ro in ("pos", "hardneg", "neutral"):
            m = (d["source"] == so) & (d["role"] == ro)
            if not m.any():
                continue
            v = pred[m].mean()
            print(f"{so + '/' + ro:24s} {int(m.sum()):7d} " + (f"{'':>8s} {v:8.1%}" if ro == "pos" else f"{v:8.1%} {'':>8s}"))
    hn = d["label"]
    worst = sorted(((pred[hn == l].mean(), l, int((hn == l).sum())) for l in np.unique(hn[d["role"] != "pos"])
                    if (hn == l).sum() >= 15), reverse=True)[:8]
    print("worst false-positive classes:", ", ".join(f"{l} {r:.0%}(n={n})" for r, l, n in worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mlp_all"); ap.add_argument("--features", choices=["all", "raw"], default="all")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--max-hn-fpr", type=float, default=0.03); ap.add_argument("--min-recall", type=float, default=0.0, help="pick the highest threshold with val recall >= this"); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="allow reusing an existing run name")
    a = ap.parse_args()
    run = C.RUNS / a.name
    if (run / "model.pt").exists() and not a.overwrite:
        sys.exit(f"{run} already exists - pick a new --name (version) or pass --overwrite")
    run.mkdir(parents=True, exist_ok=True); tee_to(run / "train.log")
    print("args:", vars(a))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = load("train"), load("val"); cols = columns(a.features)
    print(f"train {len(tr['y'])} (pos {int(tr['y'].sum())})  val {len(va['y'])} (pos {int(va['y'].sum())})  features={a.features}({len(cols)}) device={dev}")
    both = np.concatenate([tr["X"][:, cols], tr["Xm"][:, cols]])
    mean, std = both.mean(0), both.std(0) + 1e-6
    T = lambda x: torch.as_tensor((x[:, cols] - mean) / std, dtype=torch.float32, device=dev)
    Xt, Xmt, Xv = T(tr["X"]), T(tr["Xm"]), T(va["X"]); Xvm = T(va["Xm"])
    yt = torch.as_tensor(tr["y"], dtype=torch.float32, device=dev)
    w = torch.as_tensor(group_weights(tr), device=dev)
    model = MLP(len(cols)).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    neg = va["role"] != "pos"; best = (-1, None, 0); t0 = time.time()

    def val_prob():
        model.eval()
        with torch.no_grad():
            p = (torch.sigmoid(model(Xv)) + torch.sigmoid(model(Xvm))).cpu().numpy() / 2
        return p

    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(yt), device=dev); tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            flip = torch.rand(len(idx), 1, device=dev) < 0.5
            x = torch.where(flip, Xmt[idx], Xt[idx]); x = x + a.noise * torch.randn_like(x)
            # per-sample weights (group-balanced, positives and negatives carry equal total mass)
            loss = (Fn.binary_cross_entropy_with_logits(model(x), yt[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        sched.step(); p = val_prob(); ap_ = average_precision_score(va["y"], p)
        if ap_ > best[0]:
            best = (ap_, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"ep {ep:3d} loss {tot / len(perm):.4f} val AP {ap_:.4f} (best {best[0]:.4f}@{best[2]}) {time.time() - t0:.0f}s", flush=True)
        if ep - best[2] >= a.patience:
            print("early stop"); break
    model.load_state_dict(best[1]); p = val_prob()
    thr, f1, fpr, prec, rec = pick_threshold(p, va["y"], neg, a.max_hn_fpr, a.min_recall)
    print(f"\nbest epoch {best[2]}  val AP {best[0]:.4f}  thr {thr:.2f}  F1 {f1:.3f} P {prec:.3f} R {rec:.3f}  hard-neg FPR {fpr:.2%}")
    report(p, va, thr, "validation")
    out = C.model_path(a.name)
    save(out, model.cpu(), mean, std, a.features, thr,
         dict(pose_model=C.POSE_MODEL, pose_imgsz=C.POSE_IMGSZ, val_ap=best[0], val_f1=f1, hn_fpr=fpr,
              epoch=best[2], feature_names=[str(n) for n in tr["feature_names"][cols]], max_hn_fpr=a.max_hn_fpr, min_recall=a.min_recall))
    print("saved", out)


if __name__ == "__main__":
    main()
