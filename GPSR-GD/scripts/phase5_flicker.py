"""Phase 5: pick the temporal-smoothing window N on the s01/s02 clips.

    python scripts/phase5_flicker.py            # sweep N, print the table
    python scripts/phase5_flicker.py --model mlp

Runs every data/features/<clip>.npz (already-extracted 152-d vectors, real
video order) through the locked classifier, then through TemporalSmoother at a
range of N. Reports, averaged over clips:
  switches/100f  — label changes per 100 frames (flicker; lower = steadier)
  accuracy       — frames whose smoothed label == the clip's single true label
  lag(frames)    — how long after frame 0 the smoothed label first equals truth

Smoothing must cut flicker without hurting accuracy or adding much lag. Writes
results/phase5/flicker.json and prints the recommended N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from features.schema import CLASSES              # noqa: E402
from pipeline.runtime import GestureClassifier, TemporalSmoother  # noqa: E402

FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "phase5"
NS = [1, 3, 5, 7, 9, 11, 15]


def _clip_metrics(preds, true_label, n, floor):
    """preds: list of (label, conf) in frame order."""
    sm = TemporalSmoother(n=n, floor=floor)
    out = [sm.update(l, c).label for l, c in preds]
    switches = sum(a != b for a, b in zip(out, out[1:]))
    per100 = 100.0 * switches / max(1, len(out) - 1)
    acc = float(np.mean([o == true_label for o in out]))
    hit = next((i for i, o in enumerate(out) if o == true_label), len(out))
    return per100, acc, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--floor", type=float, default=None)
    args = ap.parse_args()

    clf = GestureClassifier(args.model)
    floor = args.floor if args.floor is not None else clf.floor

    clips = []
    for f in sorted(FEAT.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        true = str(d["label"][0])
        if true not in CLASSES:
            continue
        X = d["X"].astype(np.float32)
        preds = [clf.predict(v)[:2] for v in X]          # (label, conf) per frame
        clips.append((f.stem, true, preds))
    if not clips:
        raise SystemExit("no data/features/*.npz — run pipeline.extract_features")

    print(f"model={args.model}  floor={floor}  {len(clips)} clips "
          f"({sum(len(p) for _, _, p in clips)} frames)\n")
    print(f"{'N':>3s}  {'switches/100f':>13s}  {'accuracy':>9s}  {'lag(frames)':>11s}")
    rows = {}
    for n in NS:
        m = np.array([_clip_metrics(preds, true, n, floor) for _, true, preds in clips])
        sw, acc, lag = m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean()
        rows[n] = {"switches_per_100f": round(sw, 2), "accuracy": round(acc, 3),
                   "lag_frames": round(lag, 1)}
        tag = "  <- raw" if n == 1 else ""
        print(f"{n:3d}  {sw:13.2f}  {acc:9.3f}  {lag:11.1f}{tag}")

    # recommend: smallest N that cuts raw flicker >=60% and keeps accuracy within
    # 0.02 of the best seen, lag under ~half a second (15 frames @ ~30fps).
    raw_sw = rows[1]["switches_per_100f"]
    best_acc = max(r["accuracy"] for r in rows.values())
    rec = next((n for n in NS if n > 1
                and rows[n]["switches_per_100f"] <= 0.4 * raw_sw
                and rows[n]["accuracy"] >= best_acc - 0.02
                and rows[n]["lag_frames"] <= 15), 7)
    print(f"\nrecommended N = {rec}  "
          f"(flicker {rows[rec]['switches_per_100f']} vs raw {raw_sw}, "
          f"acc {rows[rec]['accuracy']}, lag {rows[rec]['lag_frames']}f)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flicker.json").write_text(json.dumps(
        {"model": args.model, "floor": floor, "clips": len(clips),
         "by_N": rows, "recommended_N": rec}, indent=2))
    print(f"-> {OUT / 'flicker.json'}")


if __name__ == "__main__":
    main()
