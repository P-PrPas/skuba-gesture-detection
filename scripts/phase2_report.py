"""Build results/phase2/dataset_report.docx — what external data we pulled, with
sample images, and how the training dataset was constructed.

    python scripts/pull_samples.py            # sample images -> data/samples_ext/
    python -m pipeline.build_dataset          # -> data/dataset/ + card
    python scripts/phase2_report.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
SAMP = ROOT / "results" / "phase2"
OUT = SAMP

SOURCES = [
    ("HaGRID v1", "cj-mills/hagrid-sample-30k-384p (HF mirror of hukenovs/hagrid)",
     "webcam / phone images, upper body, hand bbox <=16% of frame",
     "custom BY-SA-style, no NC clause (RoboCup use fine)",
     "ok, two_finger (peace), rock, thumb (like); call -> idle hard-negative"),
    ("HaGRID no_gesture", "cj-mills/hagrid-classification-512p-no-gesture-150k",
     "same, hand visible but no gesture", "as above", "idle"),
    ("HaGRID v2", "testdummyvt/hagRIDv2_512px (val split)",
     "same, 34 classes", "disputed commercial status; RoboCup use fine",
     "mini_heart (hand_heart / hand_heart2)"),
    ("COCO 2017", "person_keypoints_train2017 + images.cocodataset.org",
     "in-the-wild photos, full scenes, many sports/action shots",
     "annotations CC BY 4.0", "t_pose, raise_right_hand, raise_left_hand; + idle negatives"),
]

SAMPLE_ORDER = [
    ("ok", "HaGRID"), ("rock", "HaGRID"), ("thumb", "HaGRID"),
    ("two_finger", "HaGRID"), ("ily_negative_call", "HaGRID (-> idle negative)"),
    ("idle_hagrid", "HaGRID no_gesture -> idle"),
    ("t_pose", "COCO"), ("raise_right_hand", "COCO"), ("raise_left_hand", "COCO"),
    ("idle_coco", "COCO -> idle"),
]


def _table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for c, h in enumerate(headers):
        run = t.cell(0, c).paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            run = t.cell(r, c).paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    card = json.loads((DS / "dataset_card.json").read_text())
    pc = card["per_class"]

    doc = Document()
    doc.add_heading("SKUBA gesture detection — Phase 2 dataset", 0)
    doc.add_paragraph(f"Generated {date.today().isoformat()}").runs[0].italic = True

    doc.add_paragraph(
        "We have only 2 recorded subjects (s01: all classes; s02: laying + sit). "
        "That is too few for a leak-free subject-wise split. Decision (RoboCup@Home "
        "context, licences not a blocker): TRAIN the classifier on public datasets "
        "and keep every original s01/s02 frame as a held-out TEST set. This report "
        "is what to tell the team: which datasets, what the images look like, and "
        "how the training set was built."
    )

    doc.add_heading("1. What was downloaded", 1)
    _table(doc, ["Source", "Dataset", "Image style", "Licence", "Our classes"],
           [list(s) for s in SOURCES])
    doc.add_paragraph(
        "Nothing was downloaded in bulk. pipeline/extract_external.py streams the "
        "HuggingFace parquet shards one at a time (~50-500 MB, deleted right "
        "after), runs our MediaPipe Pose + Hands pipeline on each image, and keeps "
        "ONLY the 152-d normalised feature vector (~0.6 KB/frame). COCO downloads "
        "only the ~250 MB keypoint-annotation JSON plus the few hundred images "
        "that pass a pose filter. No source images are retained."
    )

    doc.add_heading("2. What the training images look like", 1)
    doc.add_paragraph(
        "6 examples per class below: top row = the source image, bottom row = the "
        "MediaPipe skeleton + wrist-anchored hand crop that we actually turn into "
        "features. This is exactly what the model sees."
    )
    for folder, cap in SAMPLE_ORDER:
        img = SAMP / f"sample_{folder}.jpg"
        if img.exists():
            doc.add_paragraph(f"{folder}  —  {cap}", style="Heading 3")
            doc.add_picture(str(img), width=Inches(6.0))
    doc.add_paragraph(
        "mini_heart (HaGRIDv2 hand_heart) samples could not be pulled — the "
        "HuggingFace preview API is down for that repo. Grab them on Colab if the "
        "team needs them. Note the class also failed in Phase 3: HaGRID's "
        "hand-heart is a CHEST-level finger-heart, s01's mini_heart is "
        "hands-together OVERHEAD — different pose."
    )
    doc.add_paragraph(
        "Note the COCO honesty problem visible above: COCO has ACTIONS, not "
        "gestures. 'arm raised' catches pitchers, tennis players, riders; 't_pose' "
        "catches a baby lying with arms spread. MediaPipe re-verification "
        "(section 4) drops ~55% of the COCO rows, but the survivors are still "
        "sports-flavoured. This is why t_pose / raise_hand test lower and why "
        "Phase 6 field data matters."
    )

    doc.add_heading("3. From image to feature", 1)
    for t in [
        "person -> MediaPipe Pose -> 33 body landmarks.",
        "each wrist -> square crop sized by shoulder width -> MediaPipe Hands -> "
        "21 landmarks/hand (or absent -> presence flag = 0).",
        "normalise: body centred on the shoulder-hip midpoint / scaled by "
        "shoulder-hip distance; each hand centred on its wrist / scaled by palm "
        "width (features/normalize.py).",
        "concatenate -> 152-d vector: body[0:66], left hand[66:108] + flag[108], "
        "right hand[109:151] + flag[151] (features/schema.py).",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("4. Cleaning the external features", 1)
    doc.add_paragraph(
        "pipeline/build_dataset.py::_clean_external drops: (a) rows with any "
        "|feature| > 12 — normalisation blows up when the hips are out of a webcam "
        "frame; (b) hand-shape classes (ok/rock/thumb/two_finger/mini_heart) with "
        "no hand detected; (c) COCO pose rows whose MediaPipe keypoints don't "
        "actually show the pose (COCO's own annotation frequently points at a "
        "different person in a multi-person photo)."
    )
    cs = card["external_clean"]
    _table(doc, ["source file", "raw rows", "kept", "%"],
           [[k.split("__")[-1], v["raw"], v["kept"],
             f"{100 * v['kept'] // max(1, v['raw'])}"] for k, v in cs.items()])

    doc.add_heading("5. Building train / test", 1)
    doc.add_paragraph(
        f"train.npz = {card['train']['rows']:,} rows: cleaned external features + "
        f"augmentation (mirror+relabel, limb-length jitter, rotation, keypoint "
        f"dropout, coord noise — features/augment.py). Per-class cap ~7,000; "
        f"scarce classes boosted to ~2,200."
    )
    doc.add_paragraph(
        f"test.npz = {card['test']['rows']:,} rows, NEVER augmented: 1,367 "
        f"original s01/s02 frames + 214 held-out HaGRID `thumb` (s01 has no "
        f"thumb). s01/s02 never appear in training except as augmented copies for "
        f"the 6 aug-only classes."
    )
    rows = []
    for c in card["classes"]:
        r = pc.get(c, {})
        rows.append([c, r.get("eval_type", "-"), r.get("train_source", "-"),
                     r.get("train_rows", 0), r.get("test_rows", 0)])
    _table(doc, ["class", "eval type", "train source", "train rows", "test rows"], rows)

    doc.add_heading("6. What each class's test number will mean", 1)
    for t in [
        "cross_domain (idle, ok, two_finger, rock, thumb, mini_heart, "
        "raise_L/R_hand, t_pose): external train, our test — a REAL "
        "cross-subject + cross-domain generalisation number.",
        "held_out_external (thumb): cross-subject within HaGRID, same webcam "
        "domain.",
        "aug_only_leak (sit, squat, laying, i_love_you, heart, glico_pose): no "
        "external data. Train = augmented s01 frames. For `sit`, s02's frames are "
        "still a clean cross-person check (aug pool is s01-only); for `laying` "
        "there is no s01 so it is a full leak. These are NOT generalisation "
        "numbers — Phase 6 field testing is the gate.",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("7. Known gaps", 1)
    for t in [
        "sit / laying / squat / i_love_you / heart / glico_pose — no external "
        "data found (NTU needs registration, Le2i's link is dead, ASL ILY sign "
        "is in no static-image dataset).",
        "Hand-landmark domain gap: s01's fingertip spread is ~2x HaGRID's even "
        "after palm-width normalisation (Phase 3 report). This broke `rock`.",
        "COCO poses are action photos, not held gestures.",
        "mini_heart body position (chest vs overhead).",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    out = OUT / "dataset_report.docx"
    doc.save(str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
