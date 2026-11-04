"""Fetch open training data for raise-hand detection. Never downloads a whole
dataset: Kinetics-700 / SCB zips are read with HTTP range requests (only the
selected members), HMDB51 per-file. Everything lands in data/raw/ (git-ignored)
plus a manifest CSV (clip path, source, original label, our role).

    python scripts/fetch_datasets.py k700 [--per-neg 50 --per-pos 400]
    python scripts/fetch_datasets.py hmdb51
    python scripts/fetch_datasets.py scb

`role` is the *clip-level* weak role, not a frame label:
    pos      clip is of an action that counts as raise_hand (frame labels get
             refined later by pose; see docs/datasets.md)
    hardneg  hand-near-head/face or hand-to-mouth action -> never raise_hand
    neutral  seated/standing idle-ish context -> never raise_hand
    ambig    salute etc. - needs a labelling decision, not used by default

The in-house clips in dataset/ are the TEST set and are never touched here.
"""
from __future__ import annotations

import argparse, csv, json, random, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 0

# ---------------- Kinetics-700 (HF mirror, 22 zips x ~42 GB) ----------------
K700_URL = "https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700/resolve/main/Kinetics700_part_{:03d}.zip"
K700_PARTS = 22
K700_ROLES = {
    "pos": ["waving hand"],
    "hardneg": [
        "adjusting glasses", "fixing hair", "brushing hair", "combing hair", "scrubbing face",
        "yawning", "talking on cell phone", "looking at phone", "texting", "sneezing", "coughing",
        "blowing nose", "checking watch", "winking", "rolling eyes", "drinking shots", "sipping cup",
        "tasting food", "tasting wine", "tasting beer", "eating burger", "eating hotdog",
        "eating doughnuts", "eating cake", "smoking", "putting on lipstick", "putting on mascara",
        "putting on eyeliner", "putting on foundation", "lifting hat", "using inhaler",
        "putting in contact lenses", "crying", "laughing", "fidgeting", "twiddling fingers",
        "cracking neck", "cracking knuckles", "drumming fingers", "tapping pen",
    ],
    "neutral": [
        "news anchoring", "watching tv", "reading book", "reading newspaper", "dining",
        "playing cards", "playing controller", "using remote controller (not gaming)", "writing",
        "drawing", "testifying", "answering questions", "waiting in line",
    ],
    "ambig": ["saluting"],
}


def _k700_index(tmp: Path) -> dict:
    """filename-id -> (part, member path, size). Cached; 22 central-directory reads."""
    from remotezip import RemoteZip
    cache = tmp / "index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    idx = {}
    for p in range(1, K700_PARTS + 1):
        for attempt in range(3):
            try:
                with RemoteZip(K700_URL.format(p)) as z:
                    for i in z.infolist():
                        f = i.filename
                        if f.endswith(".mp4") and not f.startswith("test/"):
                            idx[Path(f).stem[:-14]] = (p, f, i.file_size)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  part {p} retry {attempt}: {e}", file=sys.stderr); time.sleep(3)
        print(f"  indexed part {p:02d}: {len(idx)} clips so far", flush=True)
    cache.write_text(json.dumps(idx))
    return idx


def fetch_k700(per_pos: int, per_neg: int) -> None:
    from remotezip import RemoteZip
    out = RAW / "k700"; tmp = out / "_meta"; tmp.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split in ("train", "validation"):
        p = tmp / "raw" / f"{split}.json"
        if not p.exists():
            with RemoteZip(K700_URL.format(1)) as z:
                z.extract(f"raw/{split}.json", tmp)
        for vid, v in json.loads(p.read_text()).items():
            meta[vid] = v["annotations"]["label"]
    idx = _k700_index(tmp)
    rng = random.Random(SEED)
    role_of = {c: r for r, cs in K700_ROLES.items() for c in cs}
    by_cls = defaultdict(list)
    for vid, lab in meta.items():
        if lab in role_of and vid in idx:
            by_cls[lab].append(vid)
    plan = []  # (part, member, dest, row)
    for lab, vids in sorted(by_cls.items()):
        vids.sort(); rng.shuffle(vids)
        role = role_of[lab]
        n = per_pos if role == "pos" else per_neg
        print(f"  {role:8s} {lab:36s} available={len(vids):4d} take={min(n, len(vids))}")
        for vid in vids[:n]:
            part, member, size = idx[vid]
            dest = out / lab.replace(" ", "_") / Path(member).name
            plan.append((part, member, dest, dict(path=str(dest.relative_to(ROOT)), source="kinetics700",
                         label=lab, role=role, clip_id=vid, bytes=size)))
    print(f"  total {len(plan)} clips, {sum(r['bytes'] for *_, r in plan) / 1e9:.2f} GB")
    todo = defaultdict(list)
    for part, member, dest, row in plan:
        if not dest.exists():
            todo[part].append((member, dest))

    def work(part):
        n = 0
        for attempt in range(3):
            try:
                with RemoteZip(K700_URL.format(part)) as z:
                    for member, dest in todo[part]:
                        if dest.exists():
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as src, open(dest.with_suffix(".part"), "wb") as dst:
                            dst.write(src.read())
                        dest.with_suffix(".part").rename(dest); n += 1
                return part, n
            except Exception as e:  # noqa: BLE001
                print(f"  part {part} retry {attempt}: {e}", file=sys.stderr); time.sleep(5)
        return part, -1

    with ThreadPoolExecutor(6) as ex:
        for part, n in ex.map(work, sorted(todo)):
            print(f"  part {part:02d}: fetched {n}", flush=True)
    _write_manifest(out / "manifest.csv", [r for *_, r in plan if (ROOT / r["path"]).exists()])


# ---------------- HMDB51 (HF divm mirror, mp4 per clip) ----------------
HMDB_REPO = "https://huggingface.co/datasets/divm/hmdb51/resolve/main"
HMDB_ROLES = {
    "pos": ["wave"],
    "hardneg": ["brush_hair", "drink", "eat", "smoke", "chew", "talk", "laugh", "smile"],
    "neutral": ["sit", "stand", "walk", "turn"],
}


def fetch_hmdb51() -> None:
    import io, urllib.request
    out = RAW / "hmdb51"; out.mkdir(parents=True, exist_ok=True)
    role_of = {c: r for r, cs in HMDB_ROLES.items() for c in cs}
    rows = []
    for split in ("train", "validation", "test"):
        raw = urllib.request.urlopen(f"{HMDB_REPO}/{split}/metadata.csv", timeout=60).read().decode()
        for r in csv.DictReader(io.StringIO(raw)):
            if r["label"] in role_of:
                rows.append((split, r))
    print(f"  hmdb51: {len(rows)} clips in selected classes")
    man = []

    def get(item):
        split, r = item
        dest = out / r["label"] / Path(r["file_name"]).name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    data = urllib.request.urlopen(f"{HMDB_REPO}/{split}/{r['file_name'].split('/')[-1]}", timeout=120).read()
                    dest.write_bytes(data); break
                except Exception as e:  # noqa: BLE001
                    print(f"  hmdb {dest.name}: {type(e).__name__} {e}", file=sys.stderr, flush=True); time.sleep(3)
        return dict(path=str(dest.relative_to(ROOT)), source="hmdb51", label=r["label"],
                    role=role_of[r["label"]], clip_id=dest.stem, bytes=dest.stat().st_size if dest.exists() else 0)

    with ThreadPoolExecutor(8) as ex:
        man = [m for m in ex.map(get, rows) if m["bytes"] > 0]
    _write_manifest(out / "manifest.csv", man)
    print(f"  hmdb51: {len(man)} downloaded")


# ---------------- SCB-Dataset (HF wintonYF/SCB-Dataset) ----------------
SCB_ZIP = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"


def fetch_scb() -> None:
    import urllib.request
    out = RAW / "scb"; out.mkdir(parents=True, exist_ok=True)
    dest = out / "SCB5-Handrise-Read-write.zip"
    if not dest.exists():
        urllib.request.urlretrieve(SCB_ZIP, dest.with_suffix(".part")); dest.with_suffix(".part").rename(dest)
    print("  scb:", dest, f"{dest.stat().st_size / 1e6:.0f} MB")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "label", "role", "clip_id", "bytes"])
        w.writeheader(); w.writerows(rows)
    print(f"  manifest: {path} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["k700", "hmdb51", "scb"])
    ap.add_argument("--per-pos", type=int, default=400)
    ap.add_argument("--per-neg", type=int, default=50)
    a = ap.parse_args()
    {"k700": lambda: fetch_k700(a.per_pos, a.per_neg), "hmdb51": fetch_hmdb51, "scb": fetch_scb}[a.what]()
