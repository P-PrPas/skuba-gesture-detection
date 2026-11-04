"""Quick quality audit of data/raw clips: 5 frames/clip through YOLO-pose.
usable = largest person has both shoulders + >=1 wrist conf>0.5 and box height >= 35% of frame.
raised = usable and a wrist is above its same-side shoulder (the rule from probe_pose.py).
    python scripts/audit_raw.py [--model yolo11s-pose]
"""
import argparse, csv, collections
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="yolo11s-pose"); ap.add_argument("--frames", type=int, default=5)
a = ap.parse_args()
m = YOLO(str(ROOT / "weights" / f"{a.model}.pt"))
rows = [r for p in ("k700", "hmdb51") for r in csv.DictReader(open(ROOT / "data/raw" / p / "manifest.csv"))]
stat = collections.defaultdict(lambda: [0, 0, 0, 0])  # (source,role,label) -> frames, person, usable, raised
for r in rows:
    cap = cv2.VideoCapture(str(ROOT / r["path"])); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 2: continue
    frames = []
    for i in np.linspace(0.1, 0.9, a.frames) * n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: frames.append(f)
    if not frames: continue
    key = (r["source"], r["role"], r["label"])
    for f, res in zip(frames, m.predict(frames, imgsz=640, conf=0.25, verbose=False)):
        s = stat[key]; s[0] += 1
        if res.boxes is None or not len(res.boxes): continue
        b = res.boxes.xyxy.cpu().numpy(); i = int(((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])).argmax())
        s[1] += 1
        kp = res.keypoints.xy[i].cpu().numpy(); cf = res.keypoints.conf[i].cpu().numpy()
        big = (b[i, 3]-b[i, 1]) >= 0.35 * f.shape[0]
        if big and cf[5] > .5 and cf[6] > .5 and (cf[9] > .5 or cf[10] > .5):
            s[2] += 1
            if any(cf[w] > .5 and kp[w, 1] < kp[sh, 1] for w, sh in ((9, 5), (10, 6))): s[3] += 1
out = ROOT / "data/raw/audit.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["source", "role", "label", "frames", "person", "usable", "raised_rule"]); 
    for k, v in sorted(stat.items()): w.writerow([*k, *v])
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for (src, role, lab), v in stat.items():
    for j in range(4): agg[(src, role)][j] += v[j]
print(f"{'source/role':22s} {'frames':>7s} {'person':>7s} {'usable':>7s} {'raised|usable':>14s}")
for k, v in sorted(agg.items()):
    print(f"{k[0]+'/'+k[1]:22s} {v[0]:7d} {v[1]/v[0]:7.0%} {v[2]/v[0]:7.0%} {v[3]/max(v[2],1):14.0%}")
for k in [("k700", "pos", "waving hand"), ("hmdb51", "pos", "wave")]:
    v = stat[k]; print(k, f"usable={v[2]/v[0]:.0%} raised|usable={v[3]/max(v[2],1):.0%}")
print("hard-negative classes ranked by frames that look raised (these are the confusers):")
hn = sorted(((v[3]/max(v[2], 1), v[2], k[2], k[0]) for k, v in stat.items() if k[1] == "hardneg" and v[2] >= 20), reverse=True)[:8]
for r, u, lab, src in hn: print(f"   {src:7s} {lab:26s} raised|usable={r:.0%} (usable frames={u})")
