"""Build results/phase3/classifier_report.docx from the eval JSONs + dataset card.

    python -m classifier.train --model lgbm && python -m classifier.evaluate --model lgbm
    python -m classifier.train --model rf   && python -m classifier.evaluate --model rf
    python scripts/phase3_report.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"
OUT = ROOT / "results" / "phase3"

EVAL_MEANS = {
    "cross_domain": "external train, s01/s02 test — a real generalisation number",
    "held_out_external": "cross-subject within HaGRID (same webcam domain)",
    "aug_only_leak": "train = augmented s01 frames — NOT a generalisation number",
    "n/a": "-",
}


def _cell(t, r, c, text, bold=False, color=None):
    cell = t.cell(r, c)
    cell.text = ""
    run = cell.paragraphs[0].add_run(str(text))
    run.bold = bold
    run.font.size = Pt(9)
    if color:
        run.font.color.rgb = color


def _table(doc, headers, rows, colors=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for c, h in enumerate(headers):
        _cell(t, 0, c, h, bold=True)
    for r, row in enumerate(rows, 1):
        for c, v in enumerate(row):
            _cell(t, r, c, v, color=(colors[r - 1] if colors else None))
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    card = json.loads((DS / "dataset_card.json").read_text())
    per_class = card["per_class"]
    evals = {}
    for m in ("lgbm", "rf"):
        f = MODELS / f"{m}.eval.json"
        if f.exists():
            evals[m] = json.loads(f.read_text())
    if not evals:
        raise SystemExit("run classifier.evaluate first")
    primary = evals.get("rf") or next(iter(evals.values()))

    doc = Document()
    doc.add_heading("SKUBA gesture detection — Phase 3 baseline classifier", 0)
    doc.add_paragraph(f"Generated {date.today().isoformat()}").runs[0].italic = True

    v = doc.add_paragraph()
    r = v.add_run("Verdict: Phase 3 exit criterion NOT met.")
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0xB0, 0x00, 0x00)
    doc.add_paragraph(
        f"Two tree baselines (LightGBM, RandomForest) tie at macro-F1 "
        f"{primary['macro_f1_real_eval']:.2f} on the classes that are a real "
        f"generalisation test. Most classes work, but `rock` and `mini_heart` are "
        f"at 0.00 F1 — every test frame of each is classified as a different "
        f"class. Those two failures, and their causes, define the Phase 4 work; "
        f"the model-family choice does not (both trees behave the same)."
    )

    doc.add_heading("1. Architecture — where this sits", 1)
    doc.add_paragraph(
        "The pipeline has exactly ONE classifier slot: 152-d fused feature vector "
        "in -> (class, confidence) out -> temporal smoothing -> final label or "
        "idle. Phase 3 trains candidate models for that slot; it does not stack or "
        "ensemble them. LightGBM and RandomForest are two candidates. Phase 4 adds "
        "an MLP and picks one to lock (ARCHITECTURE.md 'Classifier interface')."
    )
    doc.add_paragraph(
        "LightGBM = gradient-boosted shallow trees (each corrects the last). "
        "RandomForest = many independent deep trees, averaged. Both on the raw "
        "152-d vector, class_weight='balanced', features clipped to +-10."
    )

    doc.add_heading("2. Train / test data", 1)
    doc.add_paragraph(
        f"Train: {card['train']['rows']:,} rows — cleaned external features "
        f"(HaGRID, COCO) + augmentation, for 8 classes; the other 6 "
        f"(sit, squat, laying, i_love_you, heart, glico_pose) are augmented copies "
        f"of s01 frames (no external source). See docs/phase2b_external_data.md."
    )
    doc.add_paragraph(
        f"Test: {card['test']['rows']:,} rows, never augmented — every original "
        f"s01/s02 frame (1,367) + 214 held-out HaGRID `thumb` frames. s01/s02 are "
        f"NOT in training (except as augmented copies for the 6 aug-only classes)."
    )

    doc.add_heading("3. Results", 1)
    rows, colors = [], []
    for c in card["classes"]:
        pc = primary["per_class"].get(c)
        if not pc:
            continue
        et = pc["eval_type"]
        rows.append([c, et, pc["n"], f"{pc['precision']:.2f}", f"{pc['recall']:.2f}",
                     f"{pc['f1']:.2f}", f"{pc['mean_conf']:.2f}"])
        colors.append(RGBColor(0xB0, 0, 0) if pc["f1"] == 0 else
                      (RGBColor(0x88, 0x66, 0) if pc["f1"] < 0.6 else None))
    _table(doc, ["class", "eval type", "n", "prec", "rec", "F1", "conf"], rows, colors)
    doc.add_paragraph(
        f"macro-F1: all classes {primary['macro_f1_all']:.2f}; "
        f"real-eval only {primary['macro_f1_real_eval']:.2f}. "
        "Model comparison: " + ", ".join(
            f"{m} {e['macro_f1_real_eval']:.2f}" for m, e in evals.items()) + " — a tie."
    ).runs[0].font.size = Pt(9)
    doc.add_paragraph("What each eval type means:")
    for k, txt in EVAL_MEANS.items():
        if k != "n/a":
            doc.add_paragraph(f"{k}: {txt}", style="List Bullet")

    doc.add_heading("4. sit — the one clean cross-person number", 1)
    doc.add_paragraph(
        "`sit` is trained on augmented s01 frames only, so its s02 test frames "
        "(a different person, different session, different room) are a genuine "
        "cross-person check:"
    )
    _table(doc, ["subject", "recall", "meaning"], [
        ["s01 (315 frames)", "1.00", "same person as training basis — leaked, ignore"],
        ["s02 (60 frames)", "0.77", "clean cross-person — the honest sit number"],
    ])

    doc.add_heading("5. The two failures", 1)
    doc.add_paragraph("rock -> i_love_you (all 102 frames); mini_heart -> heart (40/45).")
    for t in [
        "s01-attractor effect: `i_love_you` and `heart` are trained ONLY on "
        "augmented s01 frames, so the model over-fits that one person's hand and "
        "body. Any other s01 test frame that is geometrically nearby — s01's "
        "`rock`, s01's `mini_heart` — gets pulled in.",
        "Hand-landmark domain gap: on s01's `rock` the normalised thumb-to-pinky "
        "fingertip distance is ~1.3; on HaGRID's `rock` (what the model trained "
        "on) it is ~0.5. Palm-width normalisation does not close this — MediaPipe "
        "Hands on a tight wrist-crop of a distant robot-camera hand produces a "
        "differently proportioned skeleton than on a large frontal webcam hand.",
        "mini_heart body position: HaGRID `hand_heart` is a chest-level "
        "finger-heart; s01's `mini_heart` is hands-together overhead, which "
        "matches s01's `heart` (also overhead).",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    hg = OUT / "handgap_rock.jpg"
    if hg.exists():
        doc.add_paragraph(
            "Below: MediaPipe hand landmarks for s01 `rock` (row 1, the "
            "misclassified test frames), s01 `i_love_you` (row 2, the aug-only "
            "training basis), and HaGRID `rock` (row 3, the actual training data). "
            "Rows 1 and 2 are the same person in the same session and are nearly "
            "identical in landmark space. Row 3 shows the HaGRID wrist-crop often "
            "misses the hand entirely (crop is sized by shoulder width, which is "
            "unreliable when the hips are out of frame) — so the model's `rock` "
            "knowledge is thin to begin with."
        )
        doc.add_picture(str(hg), width=Inches(6.0))

    doc.add_heading("6. Top confusions", 1)
    conf = primary["confusion"]
    pairs = sorted(((cnt, a, b) for a, d in conf.items() for b, cnt in d.items() if a != b),
                   reverse=True)[:10]
    _table(doc, ["true", "predicted as", "count"], [[a, b, cnt] for cnt, a, b in pairs])

    doc.add_heading("7. Phase 4 plan", 1)
    for i, t in enumerate([
        "Close the hand-landmark domain gap — add hand-scale + hand-rotation "
        "jitter to features/augment.py, or canonicalise hand orientation in "
        "normalize_hand, or switch to joint-angle features (scale-free, would "
        "sidestep it entirely). IMPLEMENTATION_PLAN Phase 4 already calls for the "
        "raw-coords vs derived-features comparison.",
        "Stop `i_love_you` / `heart` poisoning neighbours — raise their per-class "
        "confidence threshold, down-weight them, or find real data (none exists "
        "for the ASL ILY handshape; HaGRID's heart is the wrong pose).",
        "mini_heart — drop the HaGRID chest-level data and treat as aug-only, or "
        "add an overhead-position augmentation.",
        "idle over-triggers (precision 0.48) — tune the idle threshold on a "
        "validation slice; the `call`->idle hard-negatives may be too aggressive.",
        "Re-run the LightGBM vs RF vs MLP comparison once the features are fixed, "
        "then lock one config (Phase 4 deliverable).",
    ], 1):
        doc.add_paragraph(f"{i}. {t}", style="List Bullet")

    out = OUT / "classifier_report.docx"
    doc.save(str(out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
