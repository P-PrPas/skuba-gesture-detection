"""Pull a handful of example images from each external training source, plus a
MediaPipe-skeleton overlay, into data/samples_ext/ for the Phase 2 report.

    python scripts/pull_samples.py

HaGRID rows come from the HF datasets-server /rows API (no parquet download).
COCO images are fetched by id from pipeline/*.done (the Colab run recorded them).
Small (~8 images/class); safe to run on the laptop.
"""

from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi

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
                "raise_left_hand": "raise_left_hand", "_coco_idle": "idle_coco"}
_COCO_IMG = "http://images.cocodataset.org/train2017/{:012d}.jpg"


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


def _save_with_overlay(img_bytes: bytes, dst: Path):
    import cv2
    import numpy as np

    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    from smoke_test import HAND_EDGES

    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), arr)

    pose = _save_with_overlay._pose
    hands = _save_with_overlay._hands
    pose.new_sequence()
    kp = pose.estimate(arr)
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


def main():
    import cv2

    from backbone.hands import HandLandmarker
    from backbone.pose import PoseEstimator

    _save_with_overlay._pose = PoseEstimator("mediapipe")
    _save_with_overlay._hands = HandLandmarker()
    OUT.mkdir(parents=True, exist_ok=True)

    for repo, split, total, by_idx, ncl in HAGRID:
        print(f"HaGRID {repo}")
        counts: dict[str, int] = {}
        for folder, img in _hf_rows(repo, split, total, by_idx, ncl):
            counts[folder] = counts.get(folder, 0) + 1
            _save_with_overlay(img, OUT / folder / f"{counts[folder]:02d}.jpg")
            print(f"  {folder} {counts[folder]}")

    # COCO by id
    done = {}
    df = ROOT / "data" / "features_ext" / "coco_pose.done"
    if df.exists():
        for line in df.read_text().splitlines():
            cls, iid = line.split(":")
            done.setdefault(cls, []).append(int(iid))
    for raw_cls, folder in COCO_CLASSES.items():
        ids = done.get(raw_cls, [])[:: max(1, len(done.get(raw_cls, [1])) // 40)][:N_PER_CLASS]
        print(f"COCO {folder}: {len(ids)} ids")
        for n, iid in enumerate(ids, 1):
            try:
                _save_with_overlay(_get(_COCO_IMG.format(iid), 40), OUT / folder / f"{n:02d}.jpg")
                print(f"  {folder} {n}")
            except Exception as e:  # noqa: BLE001
                print(f"    {iid} fail {e}")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
