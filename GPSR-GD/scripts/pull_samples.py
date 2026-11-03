"""Pull a handful of example images from each external training source, plus a
MediaPipe-skeleton overlay, into data/samples_ext/ for the Phase 2 report.

    python scripts/pull_samples.py

HaGRID rows come from the HF datasets-server /rows API (no parquet download).
COCO images are fetched by id from pipeline/*.done (the Colab run recorded them).
Small (~8 images/class); safe to run on the laptop.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "samples_ext"
N_PER_CLASS = 6
_CTX = ssl.create_default_context(cafile=certifi.where())

# HaGRID: (repo, split, approx_total, {label_index: folder}). The cj-mills
# 30k/150k samples are class-ordered so a per-class offset lands on that class;
# datasets-server /filter is 500 for these repos, /rows works.
HAGRID = [
    ("cj-mills/hagrid-sample-30k-384p", "train", 30000,
     {0: "ily_negative_call", 4: "thumb", 6: "ok", 9: "two_finger", 11: "rock"}, 18),
    ("cj-mills/hagrid-classification-512p-no-gesture-150k", "train", 150000,
     {0: "idle_hagrid"}, 1),
    ("testdummyvt/hagRIDv2_512px", "val", 100000,
     {6: "mini_heart", 7: "mini_heart"}, 34),
]
COCO_CLASSES = {"t_pose": "t_pose", "raise_right_hand": "raise_right_hand",
                "raise_left_hand": "raise_left_hand", "_coco_idle": "idle_coco",
                "sit": "sit", "laying": "laying"}
_COCO_IMG = "http://images.cocodataset.org/train2017/{:012d}.jpg"

# classes trained/tested on the s01 clips (no usable external samples) -> pull
# frames straight from data/clips/<clip>/<clip>.mp4
CLIP_CLASSES = {"squat": "squat_01", "glico_pose": "glico_pose_01",
                "mini_heart": "mini_heart_01", "sit": "sit_01", "laying": "laying_03"}


def _get(url, timeout=60):
    for attempt in range(6):
        try:
            return urllib.request.urlopen(url, timeout=timeout, context=_CTX).read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                wait = 15 * (attempt + 1)
                print(f"    429; wait {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def _posture_ok(kp, cls: str) -> bool:
    """Stricter than build_dataset._clean_external — for the report we want
    unambiguous examples, not just geometrically-plausible ones."""
    from features.normalize import normalize_body

    b = normalize_body(kp.xy)[:33]
    sho = (b[11] + b[12]) / 2
    hip = (b[23] + b[24]) / 2
    knee_dy = (b[25, 1] + b[26, 1]) / 2 - hip[1]           # + = knee below hip
    ank_dy = (b[27, 1] + b[28, 1]) / 2 - (b[25, 1] + b[26, 1]) / 2  # + = ankle below knee
    knees_level = abs(b[25, 1] - b[26, 1]) < 0.4           # both legs doing the same thing
    spine_h = abs(sho[0] - hip[0]) > abs(sho[1] - hip[1])
    if cls == "laying":
        return bool(spine_h and abs(sho[0] - hip[0]) > 1.6 * abs(sho[1] - hip[1]))
    if cls == "sit":
        return bool(not spine_h and sho[1] < hip[1] and knees_level
                    and -0.15 < knee_dy < 0.75 and ank_dy > 0.35)
    if cls == "squat":
        return bool(not spine_h and sho[1] < hip[1] and knees_level
                    and knee_dy < 0.3 and ank_dy > 0.15)
    return True


def _save_with_overlay(img_bytes: bytes, dst: Path, verify_cls: str | None = None):
    import cv2
    import numpy as np

    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    from smoke_test import HAND_EDGES

    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return False

    pose = _save_with_overlay._pose
    hands = _save_with_overlay._hands
    pose.new_sequence()
    kp = pose.estimate(arr)
    if verify_cls and (kp is None or not _posture_ok(kp, verify_cls)):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), arr)
    ov = arr.copy()
    if kp is not None:
        for i, j in pose.edges:
            if kp.visibility[i] > 0.3 and kp.visibility[j] > 0.3:
                cv2.line(ov, tuple(kp.xy[i].astype(int)), tuple(kp.xy[j].astype(int)), (0, 255, 0), 2)
        for (x, y), v in zip(kp.xy, kp.visibility):
            if v > 0.3:
                cv2.circle(ov, (int(x), int(y)), 3, (0, 165, 255), -1)
        scale = kp.shoulder_width or 80.0
        for wr in (kp.left_wrist, kp.right_wrist):
            x0, y0, x1, y1 = wrist_crop_box(wr, scale, arr.shape)
            crop = arr[y0:y1, x0:x1]
            lm = hands.detect(crop)
            if lm is not None:
                cv2.rectangle(ov, (x0, y0), (x1, y1), (200, 200, 200), 1)
                for a, b in HAND_EDGES:
                    cv2.line(ov, (int(lm[a][0] + x0), int(lm[a][1] + y0)),
                             (int(lm[b][0] + x0), int(lm[b][1] + y0)), (255, 0, 255), 1)
    cv2.imwrite(str(dst.with_name(dst.stem + "_pose.jpg")), ov)
    return True


def _hf_rows(repo: str, split: str, total: int, by_idx: dict, n_classes: int):
    """Yield (folder, image_bytes). Scans a small window at each class's expected
    offset (repo is class-ordered) and keeps rows whose label actually matches."""
    per = total // n_classes
    seen: dict[str, int] = {}
    for idx, folder in by_idx.items():
        for probe in (idx * per + per, idx * per + per + per // 2, (idx + 2) * per):
            if seen.get(folder, 0) >= N_PER_CLASS:
                break
            try:
                j = json.loads(_get(
                    f"https://datasets-server.huggingface.co/rows?dataset={repo}"
                    f"&config=default&split={split}&offset={max(0, probe)}&length=90"))
                time.sleep(3)
            except Exception as e:  # noqa: BLE001
                print(f"    {folder}@{probe}: query failed {e}")
                continue
            for row in j.get("rows", []):
                if seen.get(folder, 0) >= N_PER_CLASS:
                    break
                r = row["row"]
                if r.get("label") != idx:
                    continue
                src = r["image"]["src"] if isinstance(r["image"], dict) else r["image"]
                try:
                    yield folder, _get(src)
                    seen[folder] = seen.get(folder, 0) + 1
                    time.sleep(1)
                except Exception as e:  # noqa: BLE001
                    print(f"    img fail {e}")


def _frames_from_clip(clip_id: str, n: int):
    """Evenly-spaced frames from a data/clips/<class>/<clip_id>.mp4."""
    import cv2

    hits = list((ROOT / "data" / "clips").glob(f"*/{clip_id}.mp4"))
    if not hits:
        print(f"    no clip {clip_id}.mp4")
        return
    cap = cv2.VideoCapture(str(hits[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    want = {int(i) for i in np.linspace(total * 0.15, total * 0.85, n)}
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in want:
            ok2, buf = cv2.imencode(".jpg", fr)
            if ok2:
                yield buf.tobytes()
        fi += 1
    cap.release()


def main():
    from backbone.hands import HandLandmarker
    from backbone.pose import PoseEstimator

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="only these output folders (skip the rest)")
    ap.add_argument("--no-hf", action="store_true",
                    help="skip the HuggingFace HaGRID pulls (they rate-limit / 500)")
    args = ap.parse_args()
    pick = set(args.only) if args.only else None

    _save_with_overlay._pose = PoseEstimator("mediapipe")
    _save_with_overlay._hands = HandLandmarker()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.no_hf and (pick is None or pick & {f for _, _, _, bi, _ in HAGRID
                                                   for f in bi.values()}):
        for repo, split, total, by_idx, ncl in HAGRID:
            if pick and not (pick & set(by_idx.values())):
                continue
            print(f"HaGRID {repo}")
            counts: dict[str, int] = {}
            for folder, img in _hf_rows(repo, split, total, by_idx, ncl):
                if pick and folder not in pick:
                    continue
                counts[folder] = counts.get(folder, 0) + 1
                _save_with_overlay(img, OUT / folder / f"{counts[folder]:02d}.jpg")
                print(f"  {folder} {counts[folder]}")

    # COCO by id
    done: dict[str, list[int]] = {}
    df = ROOT / "data" / "features_ext" / "coco_pose.done"
    if df.exists():
        for line in df.read_text().splitlines():
            cls, iid = line.split(":")
            done.setdefault(cls, []).append(int(iid))
    verify = {"sit", "squat", "laying"}
    for raw_cls, folder in COCO_CLASSES.items():
        if pick and folder not in pick:
            continue
        for f in OUT.joinpath(folder).glob("c*.jpg"):      # clear stale COCO samples
            f.unlink()
        pool = done.get(raw_cls, [])
        ids = pool[:: max(1, len(pool) // 300)][:150]       # over-fetch; many rejected
        vc = raw_cls if raw_cls in verify else None
        print(f"COCO {folder}: trying up to {len(ids)} ids (verify={vc})")
        k = 0
        for iid in ids:
            if k >= N_PER_CLASS:
                break
            try:
                if _save_with_overlay(_get(_COCO_IMG.format(iid), 40),
                                      OUT / folder / f"c{k + 1:02d}.jpg", verify_cls=vc):
                    k += 1
                    print(f"  {folder} c{k}")
            except Exception as e:  # noqa: BLE001
                print(f"    {iid} fail {e}")

    # s01 clip frames for the classes with no external samples
    for folder, clip_id in CLIP_CLASSES.items():
        if pick and folder not in pick:
            continue
        print(f"clip {folder} <- {clip_id}")
        for k, buf in enumerate(_frames_from_clip(clip_id, N_PER_CLASS), 1):
            _save_with_overlay(buf, OUT / folder / f"s01_{k:02d}.jpg")
            print(f"  {folder} s01_{k}")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
