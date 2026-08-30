"""Phase 2b: pull external image datasets, extract our 152-d features, keep only
the features. Disk-safe — processes one parquet shard at a time and deletes it.

    python -m pipeline.extract_external hagrid          # HaGRID hand gestures
    python -m pipeline.extract_external hagrid_nogesture
    python -m pipeline.extract_external --list

Writes data/features_ext/<source>__<class>.npz with X (N,152), subject_id
(one id per source image — HaGRID images are ~1 unique person each), label,
source. Re-run is resumable: shards recorded in data/features_ext/<source>.done.

COCO / video sources are added as separate SOURCES entries later.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "data" / "features_ext"
TMP = ROOT / "data" / "_ext_tmp"

# label_map: HaGRID label name (or index name) -> our schema class. Unmapped rows skipped.
SOURCES = {
    "hagrid": {
        "repo": "cj-mills/hagrid-sample-30k-384p",
        "label_map": {
            "train_val_ok": "ok",
            "train_val_peace": "two_finger",
            "train_val_peace_inverted": "two_finger",
            "train_val_two_up": "two_finger",
            "train_val_rock": "rock",
            "train_val_like": "thumb",
            "train_val_call": "_ily_negative",  # shaka - hard negative for i_love_you
        },
        "max_per_class": 2500,
    },
    "hagrid_nogesture": {
        "repo": "cj-mills/hagrid-classification-512p-no-gesture-150k",
        "label_map": {"no_gesture": "idle", "train_val_no_gesture": "idle"},
        "max_per_class": 3000,
    },
}


def _get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry {attempt + 1} ({e})")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _parquet_shards(repo: str) -> list[str]:
    api = f"https://huggingface.co/api/datasets/{repo}/parquet"
    j = json.loads(_get(api, 30))
    return j["default"]["train"]


def _label_names(repo: str) -> list[str] | None:
    try:
        j = json.loads(_get(
            f"https://datasets-server.huggingface.co/first-rows?dataset={repo}"
            f"&config=default&split=train", 30))
        for f in j["features"]:
            if f["name"] == "label" and f["type"].get("names"):
                return f["type"]["names"]
    except Exception:  # noqa: BLE001
        pass
    return None


def run(source: str):
    import cv2
    import pyarrow.parquet as pq

    from backbone.hands import HandLandmarker, wrist_crop_box
    from backbone.pose import PoseEstimator
    from features.fuse import fuse
    from features.normalize import normalize_body, normalize_hand
    from features.schema import CLASSES

    cfg = SOURCES[source]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    done_file = OUT_DIR / f"{source}.done"
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    names = _label_names(cfg["repo"])
    shards = _parquet_shards(cfg["repo"])
    print(f"{source}: {cfg['repo']}  {len(shards)} shards  label_map={cfg['label_map']}")

    pose = PoseEstimator("mediapipe")
    hands = HandLandmarker()

    # accumulate per our-class
    buf: dict[str, list] = {}
    counts: dict[str, int] = {}
    existing = {}
    for cls in set(cfg["label_map"].values()):
        f = OUT_DIR / f"{source}__{cls}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=False)
            existing[cls] = [d["X"], d["subject_id"]]
            counts[cls] = int(d["X"].shape[0])
        else:
            counts[cls] = 0

    for si, url in enumerate(shards):
        tag = f"shard{si}"
        if tag in done:
            continue
        if all(counts.get(c, 0) >= cfg["max_per_class"] for c in set(cfg["label_map"].values())):
            print("  all classes full - stopping")
            break
        t0 = time.time()
        raw = _get(url, 300)
        tbl = pq.read_table(io.BytesIO(raw))
        del raw
        rows = tbl.to_pylist()
        del tbl
        kept = 0
        for r in rows:
            lbl = r["label"]
            lname = names[lbl] if (names and isinstance(lbl, int)) else str(lbl)
            cls = cfg["label_map"].get(lname)
            if cls is None or counts.get(cls, 0) >= cfg["max_per_class"]:
                continue
            img_bytes = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            pose.new_sequence()
            kp = pose.estimate(arr)
            if kp is None:
                continue
            body = normalize_body(kp.xy)
            scale = kp.shoulder_width or 80.0
            hnorm = []
            for wrist in (kp.left_wrist, kp.right_wrist):
                x0, y0, x1, y1 = wrist_crop_box(wrist, scale, arr.shape)
                crop = arr[y0:y1, x0:x1]
                lm = hands.detect(crop)
                hnorm.append(normalize_hand(lm) if lm is not None else None)
            vec = fuse(body, hnorm[0], hnorm[1])
            buf.setdefault(cls, []).append((vec, f"{source}_{counts[cls]:06d}"))
            counts[cls] = counts.get(cls, 0) + 1
            kept += 1
        done.add(tag)
        done_file.write_text("\n".join(sorted(done)))
        _flush(source, buf, existing)
        print(f"  {tag}: {len(rows)} rows -> +{kept}  "
              f"counts={ {c: counts[c] for c in sorted(counts)} }  ({time.time() - t0:.0f}s)")

    _flush(source, buf, existing, final=True)
    total = sum(counts.values())
    print(f"\n{source}: {total} feature rows -> {OUT_DIR}")
    for cls in sorted(counts):
        real = cls if cls in CLASSES else f"{cls} (aux)"
        print(f"  {real:22s} {counts[cls]}")


def _flush(source, buf, existing, final=False):
    for cls, items in list(buf.items()):
        if not items:
            continue
        X = np.stack([v for v, _ in items]).astype(np.float32)
        sid = np.array([s for _, s in items])
        if cls in existing:
            X = np.concatenate([existing[cls][0], X])
            sid = np.concatenate([existing[cls][1], sid])
        existing[cls] = [X, sid]
        np.savez_compressed(
            OUT_DIR / f"{source}__{cls}.npz",
            X=X, subject_id=sid,
            label=np.array([cls] * len(X)), source=np.array([source] * len(X)),
        )
        buf[cls] = []


def main():
    if "--list" in sys.argv:
        for k, v in SOURCES.items():
            print(f"{k:20s} {v['repo']}  -> {sorted(set(v['label_map'].values()))}")
        return
    src = sys.argv[1] if len(sys.argv) > 1 else ""
    if src not in SOURCES:
        raise SystemExit(f"unknown source {src!r}; --list to see options")
    run(src)


if __name__ == "__main__":
    main()
