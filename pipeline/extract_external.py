"""Phase 2b: pull external datasets, extract our 152-d features, keep only the
features. Disk-safe — one parquet shard (or one image) at a time, deleted after.

    python -m pipeline.extract_external --list
    python -m pipeline.extract_external hagrid            # HaGRID v1 hand gestures
    python -m pipeline.extract_external hagrid_nogesture  # idle
    python -m pipeline.extract_external hagrid_v2_heart   # mini_heart
    python -m pipeline.extract_external coco_pose         # t_pose, raise_*_hand, heart, idle
    python -m pipeline.extract_external roboflow_ily      # i_love_you  (needs ROBOFLOW_API_KEY)

**Run on Colab / a box with >=8 GB free RAM** — the SKUBA laptop OOMs on a
0.5 GB parquet shard + MediaPipe. See docs/run_extraction_elsewhere.md. Output
.npz files are tiny; commit them back, everything downstream runs on the laptop.

It NEVER downloads a whole dataset. HF datasets are read shard-by-shard over
HTTP (~50-500 MB each, deleted immediately); only images whose label maps to one
of our classes are kept. COCO downloads only the annotation JSON + the few
hundred images that pass the pose filter.
"""

from __future__ import annotations

import json
import os
import shutil
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

_HAGRID_V1 = {
    "train_val_ok": "ok", "train_val_peace": "two_finger",
    "train_val_peace_inverted": "two_finger", "train_val_two_up": "two_finger",
    "train_val_rock": "rock", "train_val_like": "thumb",
    "train_val_call": "_ily_negative",  # shaka - hard negative for i_love_you
}

# HF datasets exposing image + integer/string label as sharded parquet.
SOURCES = {
    "hagrid": {
        "kind": "hf_parquet", "repo": "cj-mills/hagrid-sample-30k-384p",
        "label_map": _HAGRID_V1, "max_per_class": 2500,
    },
    "hagrid_250k": {
        "kind": "hf_parquet", "repo": "cj-mills/hagrid-sample-250k-384p",
        "label_map": _HAGRID_V1, "max_per_class": 4000,
    },
    "hagrid_nogesture": {
        "kind": "hf_parquet", "repo": "cj-mills/hagrid-classification-512p-no-gesture-150k",
        "label_map": {"no_gesture": "idle", "train_val_no_gesture": "idle"},
        "max_per_class": 3000,
    },
    "hagrid_v2_heart": {
        "kind": "hf_parquet", "repo": "testdummyvt/hagRIDv2_512px",
        # not class-ordered; scan the smaller val split until mini_heart is full
        "shards": [f"data/val/validation-{i:05d}-of-00184.parquet" for i in range(184)],
        "label_map": {"hand_heart": "mini_heart", "hand_heart2": "mini_heart"},
        "max_per_class": 1800,
    },
    "coco_pose": {"kind": "coco", "max_per_class": 1500},
    "roboflow_ily": {
        "kind": "roboflow",
        # (workspace, project, version|None=probe, export_format, {rf_class: our_class})
        # class names are matched loosely (case / spaces / _ ignored). All five
        # verified 2026-09-01: v1 / coco / an "I Love You" category.
        "projects": [
            ("ece496-public-asl", "ece496-public-asl", 1, "coco", {"iloveyou": "i_love_you"}),
            ("vraj-atah9", "signlanguage-f0irs", 1, "coco", {"iloveyou": "i_love_you"}),
            ("actions", "actions-zqpb1", 1, "coco", {"iloveyou": "i_love_you"}),
            ("asl-auyfj", "asl-detection-lvx6a", 1, "coco", {"iloveyou": "i_love_you"}),
            ("signlanguageassistant", "signlanguageai", 1, "coco", {"iloveyou": "i_love_you"}),
        ],
        "max_per_class": 2500,
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


def _download(url: str, dest: Path, timeout: int = 600) -> None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry {attempt + 1} ({e})")
            time.sleep(3 * (attempt + 1))


def _hf_shards(cfg: dict) -> list[str]:
    repo = cfg["repo"]
    if "shards" in cfg:
        return [f"https://huggingface.co/datasets/{repo}/resolve/main/{s}" for s in cfg["shards"]]
    j = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}/parquet", 30))
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


class Extractor:
    """MediaPipe pose+hands -> our normalized 152-d vector for one BGR image."""

    def __init__(self):
        from backbone.hands import HandLandmarker, wrist_crop_box
        from backbone.pose import PoseEstimator

        self.pose = PoseEstimator("mediapipe")
        self.hands = HandLandmarker()
        self._crop = wrist_crop_box

    def __call__(self, arr) -> np.ndarray | None:
        from features.fuse import fuse
        from features.normalize import normalize_body, normalize_hand

        self.pose.new_sequence()
        kp = self.pose.estimate(arr)
        if kp is None:
            return None
        body = normalize_body(kp.xy)
        scale = kp.shoulder_width or 80.0
        hnorm = []
        for wrist in (kp.left_wrist, kp.right_wrist):
            x0, y0, x1, y1 = self._crop(wrist, scale, arr.shape)
            crop = arr[y0:y1, x0:x1]
            lm = self.hands.detect(crop)
            hnorm.append(normalize_hand(lm) if lm is not None else None)
        return fuse(body, hnorm[0], hnorm[1])


class Sink:
    """Per-class .npz accumulator with resume."""

    def __init__(self, source: str, classes: set[str]):
        self.source = source
        self.done_file = OUT_DIR / f"{source}.done"
        self.done = set(self.done_file.read_text().splitlines()) if self.done_file.exists() else set()
        self.store: dict[str, list] = {}
        self.counts: dict[str, int] = {}
        for c in classes:
            f = OUT_DIR / f"{source}__{c}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=False)
                self.store[c] = [list(d["X"]), list(d["subject_id"])]
                self.counts[c] = len(d["X"])
            else:
                self.store[c] = [[], []]
                self.counts[c] = 0

    def add(self, cls: str, vec: np.ndarray):
        self.store[cls][0].append(vec)
        self.store[cls][1].append(f"{self.source}_{self.counts[cls]:06d}")
        self.counts[cls] += 1

    def full(self, cap: int) -> bool:
        return all(v >= cap for v in self.counts.values())

    def flush(self, shard_tag: str | None = None):
        for c, (xs, sids) in self.store.items():
            if not xs:
                continue
            X = np.stack(xs).astype(np.float32)
            np.savez_compressed(
                OUT_DIR / f"{self.source}__{c}.npz",
                X=X, subject_id=np.array(sids),
                label=np.array([c] * len(X)), source=np.array([self.source] * len(X)),
            )
        if shard_tag:
            self.done.add(shard_tag)
            self.done_file.write_text("\n".join(sorted(self.done)))


def run_hf_parquet(source: str):
    import cv2
    import pyarrow.parquet as pq

    cfg = SOURCES[source]
    names = _label_names(cfg["repo"])
    shards = _hf_shards(cfg)
    lmap, cap = cfg["label_map"], cfg["max_per_class"]
    classes = set(lmap.values())
    print(f"{source}: {cfg['repo']}  {len(shards)} shards  -> {sorted(classes)}", flush=True)

    ex = Extractor()
    sink = Sink(source, classes)

    for si, url in enumerate(shards):
        tag = f"shard{si}"
        if tag in sink.done:
            continue
        if sink.full(cap):
            print("  all classes full - stopping")
            break
        t0 = time.time()
        p = TMP / f"{source}_{si}.parquet"
        try:
            _download(url, p)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}: download failed ({e}); skipping")
            continue
        kept = seen = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                seen += 1
                lbl = r["label"]
                lname = names[lbl] if (names and isinstance(lbl, int)) else str(lbl)
                cls = lmap.get(lname)
                if cls is None or sink.counts[cls] >= cap:
                    continue
                ib = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
                arr = cv2.imdecode(np.frombuffer(ib, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                vec = ex(arr)
                if vec is not None:
                    sink.add(cls, vec)
                    kept += 1
        p.unlink(missing_ok=True)
        sink.flush(tag)
        print(f"  {tag}: {seen} rows -> +{kept}  {dict(sorted(sink.counts.items()))}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    sink.flush()
    print(f"\n{source}: {sum(sink.counts.values())} rows -> {OUT_DIR}")
    for c in sorted(sink.counts):
        print(f"  {c:22s} {sink.counts[c]}")


# ---------------- COCO person-keypoints -> t_pose / raise_right_hand / idle ----------------
_COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_COCO_IMG = "http://images.cocodataset.org/train2017/{:012d}.jpg"
# COCO 17-kpt: 0 nose 5 Lsho 6 Rsho 7 Lelb 8 Relb 9 Lwri 10 Rwri 11 Lhip 12 Rhip 15 Lank 16 Rank


def _coco_kp(ann):
    a = np.array(ann["keypoints"], np.float32).reshape(17, 3)  # x, y, v
    return a[:, :2], a[:, 2]


def _classify_coco(xy, v) -> str | None:
    def ok(*i):
        return all(v[k] >= 1 for k in i)
    nose, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip = (
        xy[j] for j in (0, 5, 6, 7, 8, 9, 10, 11, 12))
    leye, reye = xy[1], xy[2]
    sho_y = (lsho[1] + rsho[1]) / 2
    sho_w = abs(lsho[0] - rsho[0]) + 1e-6
    if ok(0, 1, 2, 5, 6, 7, 8, 9, 10):
        # heart: both wrists above the eyes, hands meeting near the midline, both
        # elbows bowed outboard of the shoulders (the overhead two-arm heart).
        eye_y = min(leye[1], reye[1])
        overhead = lwri[1] < eye_y and rwri[1] < eye_y
        together = abs(lwri[0] - rwri[0]) < 1.1 * sho_w
        elbows_out = lelb[0] > lsho[0] and relb[0] < rsho[0]
        if overhead and together and elbows_out:
            return "heart"
    if ok(5, 6, 9, 10, 0):
        # t_pose: both wrists near shoulder height, spread wide, standing
        near = abs(lwri[1] - sho_y) < 0.35 * sho_w and abs(rwri[1] - sho_y) < 0.35 * sho_w
        wide = abs(lwri[0] - rwri[0]) > 2.2 * sho_w
        if near and wide:
            return "t_pose"
    if ok(10, 0, 6):
        # raise_right_hand: person's right wrist above the nose, left hand not raised
        left_low = (not ok(9)) or lwri[1] > lsho[1]
        if rwri[1] < nose[1] and left_low:
            return "raise_right_hand"
    if ok(9, 0, 5):
        left_up = lwri[1] < nose[1]
        right_low = (not ok(10)) or rwri[1] > rsho[1]
        if left_up and right_low:
            return "raise_left_hand"
    if ok(5, 6, 9, 10, 11, 12):
        # idle negative: hands below shoulders, roughly upright, arms down
        hands_down = lwri[1] > sho_y + 0.5 * sho_w and rwri[1] > sho_y + 0.5 * sho_w
        if hands_down:
            return "_coco_idle"
    return None


def run_coco(source: str = "coco_pose"):
    import zipfile

    import cv2

    cap = SOURCES[source]["max_per_class"]
    TMP.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_zip = TMP / "coco_ann.zip"
    ann_json = TMP / "person_keypoints_train2017.json"
    if not ann_json.exists():
        print("downloading COCO annotations (~250 MB)...", flush=True)
        _download(_COCO_ANN, ann_zip)
        with zipfile.ZipFile(ann_zip) as z:
            with z.open("annotations/person_keypoints_train2017.json") as s, open(ann_json, "wb") as d:
                while chunk := s.read(1 << 20):
                    d.write(chunk)
        ann_zip.unlink()

    print("scanning annotations...", flush=True)
    data = json.loads(ann_json.read_text())
    want: dict[str, list[int]] = {}
    for ann in data["annotations"]:
        if ann.get("num_keypoints", 0) < 8 or ann.get("iscrowd"):
            continue
        xy, vis = _coco_kp(ann)
        cls = _classify_coco(xy, vis)
        if cls:
            want.setdefault(cls, []).append(ann["image_id"])
    # bound the download loop: shuffle + keep ~3x the cap per class (MediaPipe
    # drops some, and a raised-hand pose is common in COCO -> thousands of hits).
    rng = np.random.default_rng(0)
    for c in want:
        ids = sorted(set(want[c]))
        rng.shuffle(ids)
        want[c] = ids[: SOURCES[source]["max_per_class"] * 3]
    for c, ids in want.items():
        print(f"  candidate {c}: {len(ids)} images (capped)")

    classes = {"t_pose", "raise_right_hand", "raise_left_hand", "_coco_idle", "heart"}
    ex = Extractor()
    sink = Sink(source, classes)
    # do the scarce classes first so an interrupt keeps the valuable ones
    order = sorted(want, key=lambda c: len(want[c]))
    for cls in order:
        ids = want[cls]
        if sink.counts[cls] >= cap:
            print(f"  {cls}: already {sink.counts[cls]} - skip", flush=True)
            continue
        t0, tried = time.time(), 0
        for iid in ids:
            if sink.counts[cls] >= cap:
                break
            tag = f"{cls}:{iid}"
            if tag in sink.done:
                continue
            tried += 1
            p = TMP / f"coco_{iid}.jpg"
            try:
                _download(_COCO_IMG.format(iid), p, 60)
                arr = cv2.imread(str(p))
            except Exception:  # noqa: BLE001
                arr = None
            p.unlink(missing_ok=True)
            sink.done.add(tag)
            if arr is None:
                continue
            vec = ex(arr)
            if vec is not None:
                sink.add(cls, vec)
            if tried % 200 == 0:
                sink.flush()
                sink.done_file.write_text("\n".join(sorted(sink.done)))
                print(f"    {cls}: {sink.counts[cls]}/{cap} kept, {tried} tried "
                      f"({time.time() - t0:.0f}s)", flush=True)
        sink.flush()
        sink.done_file.write_text("\n".join(sorted(sink.done)))
        print(f"  {cls}: {sink.counts[cls]} kept ({tried} tried, {time.time() - t0:.0f}s)", flush=True)
    print(f"\n{source}: {sum(sink.counts.values())} rows -> {OUT_DIR}")


# ---------------- Roboflow Universe -> i_love_you (ASL ILY handshape) ----------------
# Roboflow needs a free API key: roboflow.com -> account -> Settings -> API ->
# "Private API Key". Set it as ROBOFLOW_API_KEY. `pip install roboflow`.


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _iter_roboflow_export(loc: Path, lmap: dict):
    """Yield (image_path, bbox_or_None, our_class) from a downloaded export,
    handling both COCO-detection and folder-classification layouts. Class names
    are matched loosely via _norm()."""
    nmap = {_norm(k): v for k, v in lmap.items()}
    for d in [loc, *loc.iterdir()] if loc.is_dir() else []:
        if not d.is_dir():
            continue
        annf = d / "_annotations.coco.json"
        if annf.exists():                                   # detection export
            j = json.loads(annf.read_text())
            cats = {c["id"]: nmap.get(_norm(c["name"])) for c in j["categories"]}
            per_img: dict[int, list] = {}
            for a in j["annotations"]:
                per_img.setdefault(a["image_id"], []).append(a)
            for im in j["images"]:
                for a in per_img.get(im["id"], []):
                    cls = cats.get(a["category_id"])
                    if cls:
                        yield d / im["file_name"], a["bbox"], cls
                        break
        else:                                               # classification export
            for sub in d.iterdir():
                cls = nmap.get(_norm(sub.name)) if sub.is_dir() else None
                if not cls:
                    continue
                for imgf in sorted(sub.glob("*.jpg")) + sorted(sub.glob("*.png")):
                    yield imgf, None, cls


def run_roboflow(source: str):
    import cv2
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY — free key at roboflow.com "
                         "(account -> Settings -> API -> Private API Key)")
    cfg = SOURCES[source]
    cap = cfg["max_per_class"]
    classes = {c for *_, m in cfg["projects"] for c in m.values()}
    print(f"{source}: {len(cfg['projects'])} Roboflow projects -> {sorted(classes)}", flush=True)

    rf = Roboflow(api_key=key)
    ex = Extractor()
    sink = Sink(source, classes)

    for ws, proj, ver, fmt, lmap in cfg["projects"]:
        tag = f"{ws}/{proj}"
        if tag in sink.done:
            continue
        if sink.full(cap):
            print("  all classes full - stopping")
            break
        loc = TMP / f"rf_{proj}"
        if loc.exists():
            shutil.rmtree(loc, ignore_errors=True)
        t0 = time.time()
        project = rf.workspace(ws).project(proj)
        # .versions() is unreliable for Universe projects you don't own -> just
        # try candidate version numbers / formats and take the first that
        # downloads actual images.
        fmts = list(dict.fromkeys([fmt, "coco", "folder", "multiclass"]))
        vers = [ver] if ver else list(range(1, 9))
        got = False
        for cand, f in ((v, f) for v in vers for f in fmts):
            try:
                project.version(cand).download(f, location=str(loc))
            except Exception:  # noqa: BLE001
                shutil.rmtree(loc, ignore_errors=True)
                continue
            if any(loc.rglob("*.jpg")) or any(loc.rglob("*.png")):
                got = True
                print(f"  {tag}: version {cand}, format {f}", flush=True)
                break
            shutil.rmtree(loc, ignore_errors=True)
        if not got:
            print(f"  {tag}: nothing downloaded (versions 1-12, formats {fmts}) - skipping",
                  flush=True)
            continue
        kept = seen = 0
        # the bbox only tells us the image *contains* an ILY hand — feed the WHOLE
        # frame to the extractor so MediaPipe can find the body and do its own
        # wrist-anchored hand crop (cropping to the bbox here kills the body).
        for imgf, _bbox, cls in _iter_roboflow_export(loc, lmap):
            if sink.counts[cls] >= cap:
                continue
            seen += 1
            arr = cv2.imread(str(imgf))
            if arr is None:
                continue
            vec = ex(arr)
            if vec is not None:
                sink.add(cls, vec)
                kept += 1
        shutil.rmtree(loc, ignore_errors=True)
        sink.flush(tag)
        print(f"  {tag}: {seen} ILY images -> +{kept} kept  "
              f"{dict(sorted(sink.counts.items()))}  ({time.time() - t0:.0f}s)", flush=True)
    sink.flush()
    print(f"\n{source}: {sum(sink.counts.values())} rows -> {OUT_DIR}")
    for c in sorted(sink.counts):
        print(f"  {c:22s} {sink.counts[c]}")


def main():
    if "--list" in sys.argv or len(sys.argv) < 2:
        for k, v in SOURCES.items():
            if "label_map" in v:
                cls = sorted(set(v["label_map"].values()))
            elif "projects" in v:
                cls = sorted({c for *_, m in v["projects"] for c in m.values()})
            else:
                cls = "[coco filter]"
            print(f"{k:20s} {v.get('repo', v['kind']):45s} -> {cls}")
        return
    src = sys.argv[1]
    if src not in SOURCES:
        raise SystemExit(f"unknown source {src!r}; --list to see options")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    runner = {"coco": run_coco, "roboflow": run_roboflow}.get(
        SOURCES[src]["kind"], run_hf_parquet)
    runner(src)


if __name__ == "__main__":
    main()
