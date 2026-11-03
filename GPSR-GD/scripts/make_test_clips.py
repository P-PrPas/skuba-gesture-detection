"""Build derived test clips from the recorded material — for exercising the
Phase 5 pipeline against known actions.

    python scripts/make_test_clips.py

Outputs (git-ignored under data/, regenerate as needed):
  data/main_actions.mp4          main.MOV + main3.MOV concatenated — one clip
                                 covering idle / raise_L / raise_R / sit / squat
                                 / laying + the hand gestures. GT:
                                 data/segments_main_actions.csv
  data/2p_raise_lr.mp4           synthetic 2-person: two single-person clips side
  data/2p_sit_laying.mp4         by side (540x960 each), shorter one looped, so
  data/2p_idle_squat.mp4         each half has a known action. Tests the
                                 multi-person fan-out + per-person tracking.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS = DATA / "clips"


def concat_main_actions():
    a, b = DATA / "_a.mp4", DATA / "_b.mp4"
    for src, dst in ((DATA / "main.MOV", a), (DATA / "main3.MOV", b)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-vf", "scale=1080:1920", "-c:v", "libx264", "-crf", "20",
                        "-preset", "veryfast", "-an", "-r", "30", str(dst)], check=True)
    lst = DATA / "_cc.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(DATA / "main_actions.mp4")], check=True)
    a.unlink(); b.unlink(); lst.unlink()

    off = 2275 / 30.0                                   # main.MOV duration
    rows = [["clip_id", "subject_id", "session_id", "source_video", "class",
             "start_s", "end_s", "confidence", "notes"]]
    for r in csv.DictReader((DATA / "segments.csv").open(encoding="utf-8")):
        v = r["source_video"]
        shift = 0.0 if v == "main.MOV" else off if v == "main3.MOV" else None
        if shift is None:
            continue
        rows.append([r["clip_id"], r["subject_id"], r["session_id"], "main_actions.mp4",
                     r["class"], f'{float(r["start_s"]) + shift:.2f}',
                     f'{float(r["end_s"]) + shift:.2f}', r["confidence"], r["notes"]])
    with (DATA / "segments_main_actions.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"main_actions.mp4 + segments_main_actions.csv ({len(rows) - 1} segments)")


def _frames(path, w, h):
    c = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(cv2.resize(f, (w, h)))
    c.release()
    return out


def side_by_side():
    W, H = 540, 960
    pairs = [("raise_left_hand/raise_left_hand_01", "raise_right_hand/raise_right_hand_01", "2p_raise_lr"),
             ("sit/sit_01", "laying/laying_03", "2p_sit_laying"),
             ("idle/idle_01", "squat/squat_01", "2p_idle_squat")]
    for a, b, name in pairs:
        fa, fb = _frames(CLIPS / f"{a}.mp4", W, H), _frames(CLIPS / f"{b}.mp4", W, H)
        n = max(len(fa), len(fb))
        w = cv2.VideoWriter(str(DATA / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            30, (2 * W, H))
        for i in range(n):
            w.write(np.hstack([fa[i % len(fa)], fb[i % len(fb)]]))
        w.release()
        print(f"{name}.mp4  ({n}f, L={Path(a).name} R={Path(b).name})")


if __name__ == "__main__":
    concat_main_actions()
    side_by_side()
