"""Cut a source clip into per-class sub-clips from a segments CSV.

    python scripts/cut_segments.py data/main.MOV data/segments.csv data/clips

CSV columns: clip_id,class,start_s,end_s,confidence,notes
Rows with an empty `class` go to <out>/_review/ for manual labelling.
Re-encodes (frame-accurate) and keeps the source rotation metadata.
"""

import csv
import subprocess
import sys
from pathlib import Path


def main():
    src, csv_path, out_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        cls = (r["class"] or "").strip()
        sub = cls if cls else "_review"
        dst_dir = out_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{r['clip_id']}.mp4"
        ss, to = float(r["start_s"]), float(r["end_s"])
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", src,
            "-map_metadata", "0",
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-an",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        print(f"{dst}  ({to - ss:.1f}s)  conf={r['confidence'] or '-'}")


if __name__ == "__main__":
    main()
