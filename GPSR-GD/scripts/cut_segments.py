"""Cut per-class sub-clips from a segments CSV.

    python scripts/cut_segments.py data/segments.csv data/clips [data/]

Each row's `source_video` column names its clip (resolved against the optional
source dir, default `data/`). Rows with an empty `class` go to <out>/_review/.
Re-encodes frame-accurate and bakes in the source rotation, so cv2.VideoCapture
delivers upright frames.
"""

import csv
import subprocess
import sys
from pathlib import Path


def main():
    csv_path = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    src_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        src = src_dir / r["source_video"]
        if not src.exists():
            print(f"  SKIP {r['clip_id']}: {src} not found")
            continue
        cls = (r["class"] or "").strip()
        dst_dir = out_root / (cls if cls else "_review")
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{r['clip_id']}.mp4"
        ss, to = float(r["start_s"]), float(r["end_s"])
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", str(src),
             "-map_metadata", "0",
             "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-an",
             str(dst)],
            check=True,
        )
        print(f"{dst}  [{r['source_video']} {ss:.1f}-{to:.1f}]  {r['subject_id']}  conf={r['confidence'] or '-'}")


if __name__ == "__main__":
    main()
