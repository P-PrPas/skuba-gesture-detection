"""Run YOLO-pose over every raw source and cache keypoints (resumable, one npz per clip/image).

    python -m pipeline.extract_features videos  [--fps 4] [--limit N] [--shard i/n]   # K700 + HMDB51
    python -m pipeline.extract_features scb     [--limit N]                           # SCB images
    python -m pipeline.extract_features test    # user's held-out clips, full fps + ByteTrack

Videos: frames sampled at ~--fps, largest person per frame kept (clip label refers to the main actor).
Output: data/kpts/<source>/<clip_id>.npz  (kpt [T,17,3], box [T,4], n_person [T], t [T], fps)
SCB: one npz per split holding every person (kpt, box, scb_cls, image) -- matched to labels by IoU.
Re-running skips finished items, so it is safe to interrupt.
"""
from __future__ import annotations
import argparse, csv, sys, time, zipfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def load_model(name: str):
    return YOLO(str(C.WEIGHTS / f"{name}.pt"))


def largest(res):
    """-> (kpt [17,3], box [4], n_person) for the biggest person in a ultralytics Result."""
    kp = np.zeros((17, 3), np.float32); bx = np.zeros(4, np.float32)
    if res.boxes is None or not len(res.boxes):
        return kp, bx, 0
    b = res.boxes.xyxy.cpu().numpy()
    i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())
    kp[:, :2] = res.keypoints.xy[i].cpu().numpy(); kp[:, 2] = res.keypoints.conf[i].cpu().numpy()
    return kp, b[i], len(b)


def read_sampled(path: Path, target_fps: float, max_frames: int):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / target_fps)), 1)
    frames, idx, i = [], [], 0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f); idx.append(i)
        i += 1
    cap.release()
    return frames, np.array(idx, np.int32), fps


def do_videos(a) -> None:
    model = load_model(a.model)
    rows = [r for s in ("k700", "hmdb51") for r in csv.DictReader(open(C.RAW / s / "manifest.csv"))]
    i, n = (int(x) for x in a.shard.split("/"))
    rows = rows[i::n]
    if a.limit:  # stratified-ish: interleave so a small limit still spans every class
        rng = np.random.default_rng(0); rows = [rows[j] for j in rng.permutation(len(rows))[:a.limit]]
    t0 = time.time(); done = skipped = failed = 0
    for k, r in enumerate(rows, 1):
        out = C.KPTS / r["source"] / f"{r['clip_id']}.npz"
        if out.exists():
            skipped += 1; continue
        try:
            frames, idx, fps = read_sampled(C.ROOT / r["path"], a.fps, a.max_frames)
            if not frames:
                raise RuntimeError("no frames")
            kp, bx, npers = [], [], []
            for s in range(0, len(frames), 32):
                for res in model.predict(frames[s:s + 32], imgsz=C.POSE_IMGSZ, conf=C.POSE_CONF, verbose=False):
                    q, b, m = largest(res); kp.append(q); bx.append(b); npers.append(m)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, kpt=np.stack(kp), box=np.stack(bx), n_person=np.array(npers, np.int16),
                                t=idx / fps, fps=fps, frame_h=frames[0].shape[0], frame_w=frames[0].shape[1])
            done += 1
        except Exception as e:  # keep going; report at the end
            failed += 1; print(f"  FAIL {r['path']}: {e}", flush=True)
        if k % 25 == 0 or k == len(rows):
            el = time.time() - t0
            print(f"[{k}/{len(rows)}] done={done} skipped={skipped} failed={failed} "
                  f"{el:.0f}s elapsed, ETA {el / max(done, 1) * (len(rows) - k):.0f}s", flush=True)


def do_scb(a) -> None:
    model = load_model(a.model)
    z = zipfile.ZipFile(C.RAW / "scb" / "SCB5-Handrise-Read-write.zip")
    root = "SCB5-Handrise-Read-write-2024-9-17/"
    for split in ("train", "val"):
        out = C.KPTS / "scb" / f"{split}.npz"
        if out.exists():
            print("skip", out); continue
        names = sorted(x for x in z.namelist() if x.startswith(f"{root}images/{split}/") and x.endswith(".jpg"))
        if a.limit:
            names = names[:a.limit]
        rec = {k: [] for k in ("kpt", "box", "scb_cls", "image", "img_h")}
        t0 = time.time()
        for k, nm in enumerate(names, 1):
            img = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            res = model.predict(img, imgsz=C.POSE_IMGSZ, conf=C.POSE_CONF, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0, 4))
            lab = z.read(nm.replace("/images/", "/labels/").replace(".jpg", ".txt")).decode().split("\n")
            gt = np.array([[float(v) for v in l.split()] for l in lab if l.strip()]).reshape(-1, 5)
            gt_xyxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * W, (gt[:, 2] - gt[:, 4] / 2) * H,
                                (gt[:, 1] + gt[:, 3] / 2) * W, (gt[:, 2] + gt[:, 4] / 2) * H], 1) if len(gt) else np.zeros((0, 4))
            for j in range(len(boxes)):
                cls = -1
                if len(gt_xyxy):
                    iou = _iou(boxes[j], gt_xyxy); m = int(iou.argmax())
                    if iou[m] >= 0.5:
                        cls = int(gt[m, 0])
                rec["kpt"].append(np.concatenate([res.keypoints.xy[j].cpu().numpy(), res.keypoints.conf[j].cpu().numpy()[:, None]], 1))
                rec["box"].append(boxes[j]); rec["scb_cls"].append(cls); rec["image"].append(nm.split("/")[-1]); rec["img_h"].append(H)
            if k % 200 == 0:
                print(f"  scb/{split} [{k}/{len(names)}] {time.time() - t0:.0f}s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, kpt=np.array(rec["kpt"], np.float32), box=np.array(rec["box"], np.float32),
                            scb_cls=np.array(rec["scb_cls"], np.int8), image=np.array(rec["image"]), img_h=np.array(rec["img_h"], np.int32))
        print("wrote", out, len(rec["kpt"]), "persons")


def _iou(b, g):
    x1 = np.maximum(b[0], g[:, 0]); y1 = np.maximum(b[1], g[:, 1]); x2 = np.minimum(b[2], g[:, 2]); y2 = np.minimum(b[3], g[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter + 1e-9)


def do_test(a) -> None:
    """Held-out clips: every frame, ByteTrack, largest person (single-subject clips)."""
    model = load_model(a.model)
    for clip in sorted(C.TEST_CLIPS.glob("*.MOV")) + sorted(C.TEST_CLIPS.glob("*.mp4")):
        out = C.KPTS / "test" / f"{clip.stem}.npz"
        if out.exists():
            print("skip", out); continue
        cap = cv2.VideoCapture(str(clip)); fps = cap.get(cv2.CAP_PROP_FPS)
        kp, bx, ids, npers = [], [], [], []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            res = model.track(f, persist=True, imgsz=C.POSE_IMGSZ, conf=C.POSE_CONF, verbose=False)[0]
            q, b, m = largest(res)
            tid = -1
            if m and res.boxes.id is not None:
                bb = res.boxes.xyxy.cpu().numpy(); tid = int(res.boxes.id[int(((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])).argmax())])
            kp.append(q); bx.append(b); ids.append(tid); npers.append(m)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, kpt=np.stack(kp), box=np.stack(bx), ids=np.array(ids), n_person=np.array(npers),
                            t=np.arange(len(kp)) / fps, fps=fps)
        print("wrote", out, len(kp), "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["videos", "scb", "test"])
    ap.add_argument("--model", default=C.POSE_MODEL)
    ap.add_argument("--fps", type=float, default=4.0, help="video sampling rate")
    ap.add_argument("--max-frames", type=int, default=48, help="cap on sampled frames per clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip (to run on several GPUs)")
    a = ap.parse_args()
    {"videos": do_videos, "scb": do_scb, "test": do_test}[a.what](a)
