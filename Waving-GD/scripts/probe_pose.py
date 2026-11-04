"""Run YOLO-pose over a clip, dump COCO-17 keypoints, print quality stats, and
render an overlay video + raw 'wrist above shoulder' flag for eyeballing.

    python scripts/probe_pose.py dataset/IMG_8243.MOV --model yolo11n-pose --imgsz 640

Output: outputs/probe/pose_models/<clip>__<model>.npz  (kpt [T,17,3] xy+conf, box [T,4], ids [T])
        outputs/probe/pose_models/<clip>__<model>.mp4  (overlay)
Only the largest tracked person per frame is kept (single-person probe).
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--model", default="yolo11n-pose")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    clip = Path(a.clip)
    model = YOLO(str(ROOT / "weights" / f"{a.model}.pt"))
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_dir = ROOT / "outputs" / "probe" / "pose_models"; out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{clip.stem}__{a.model}"
    writer = None
    kpts, boxes, ids, raised, times = [], [], [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model.track(frame, persist=True, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        kp = np.zeros((17, 3), np.float32); bx = np.zeros(4, np.float32); tid = -1
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            i = int(area.argmax())
            kp = np.concatenate([r.keypoints.xy[i].cpu().numpy(),
                                 r.keypoints.conf[i].cpu().numpy()[:, None]], 1)
            bx = xyxy[i]
            if r.boxes.id is not None:
                tid = int(r.boxes.id[i])
        kpts.append(kp); boxes.append(bx); ids.append(tid)

        # raw rule flag: either wrist above (smaller y than) its own-side shoulder
        up = any(kp[w, 2] > 0.3 and kp[s, 2] > 0.3 and kp[w, 1] < kp[s, 1]
                 for w, s in ((L_WR, L_SH), (R_WR, R_SH)))
        raised.append(up)

        vis = frame.copy()
        for p, q in SKELETON:
            if kp[p, 2] > 0.3 and kp[q, 2] > 0.3:
                cv2.line(vis, tuple(kp[p, :2].astype(int)), tuple(kp[q, :2].astype(int)), (0, 255, 0), 3)
        for j in range(17):
            c = (0, 0, 255) if j in (L_WR, R_WR) else (255, 255, 0)
            if kp[j, 2] > 0.3:
                cv2.circle(vis, tuple(kp[j, :2].astype(int)), 7, c, -1)
        cv2.putText(vis, "RAISE" if up else "-", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255) if up else (200, 200, 200), 6)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(vis)
    cap.release(); writer and writer.release()

    kpts = np.stack(kpts); raised = np.array(raised)
    np.savez(out_dir / f"{stem}.npz", kpt=kpts, box=np.stack(boxes), ids=np.array(ids),
             raised_rule=raised, fps=fps)
    T = len(kpts); t = np.array(times[3:]) * 1000
    print(f"{stem}: {T} frames  det={np.mean(kpts[:,:,2].max(1) > 0):.0%}  "
          f"track-ids={sorted(set(ids) - {-1})}")
    for n, j in (("L_sh", L_SH), ("R_sh", R_SH), ("L_wr", L_WR), ("R_wr", R_WR)):
        print(f"  conf {n}: mean={kpts[:, j, 2].mean():.2f}  <0.3 in {np.mean(kpts[:, j, 2] < 0.3):.0%}")
    print(f"  rule 'wrist above shoulder': {raised.mean():.0%} of frames")
    print(f"  latency: {t.mean():.1f} ms/frame (p95 {np.percentile(t, 95):.1f}) incl. tracking")


if __name__ == "__main__":
    main()
