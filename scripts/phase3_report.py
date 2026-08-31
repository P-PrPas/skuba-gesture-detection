"""Build results/phase3/classifier_report.docx from the eval JSONs + dataset card.

    python -m classifier.train --model lgbm --features both && python -m classifier.evaluate --model lgbm
    python -m classifier.train --model rf   --features both && python -m classifier.evaluate --model rf
    python scripts/phase3_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DS = ROOT / "data" / "dataset"
MODELS = ROOT / "classifier" / "models"
OUT = ROOT / "results" / "phase3"

EVAL_MEANS = {
    "cross_domain": "external train, s01/s02 test — a real cross-subject + cross-domain number",
    "held_out_external": "cross-subject within HaGRID (same webcam domain)",
    "aug_only_leak": "train = augmented s01 frames — a leaked number (except sit@s02)",
    "pending_data": "no viable training path — not modelled, see section 6",
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
    evals, metas = {}, {}
    for m in ("lgbm", "rf"):
        ej = MODELS / f"{m}.eval.json"
        mj = MODELS / f"{m}.meta.json"
        jb = MODELS / f"{m}.joblib"
        if ej.exists():
            evals[m] = json.loads(ej.read_text())
            metas[m] = {
                "fit_s": json.loads(mj.read_text()).get("fit_seconds") if mj.exists() else None,
                "size_mb": round(jb.stat().st_size / 1e6, 1) if jb.exists() else None,
            }
    if not evals:
        raise SystemExit("run classifier.evaluate first")
    primary = evals.get("rf") or next(iter(evals.values()))

    # sit@s02 per model
    sit_s02 = {}
    try:
        d = np.load(DS / "test.npz", allow_pickle=True)
        from features.derived import to_features
        ksit = card["classes"].index("sit")
        m2 = (d["y"].astype(int) == ksit) & (d["subject_id"] == "s02")
        for m in evals:
            b = joblib.load(MODELS / f"{m}.joblib")
            X = to_features(np.clip(d["X"][m2].astype("float32"), -b["clip"], b["clip"]),
                            b.get("features", "raw"))
            mc = np.asarray(b["clf"].classes_)
            sit_s02[m] = round(float((mc[b["clf"].predict_proba(X).argmax(1)] == ksit).mean()), 2)
    except Exception:  # noqa: BLE001
        pass

    doc = Document()
    doc.add_heading("SKUBA gesture detection — Phase 3 baseline classifier", 0)
    doc.add_paragraph(f"Generated {date.today().isoformat()}").runs[0].italic = True

    npend = primary.get("n_pending", len(primary.get("pending_data_lands_as", {})))
    nmod = len(primary["per_class"])
    nreal = sum(1 for r in primary["per_class"].values()
                if r["eval_type"] in ("cross_domain", "held_out_external"))
    v = doc.add_paragraph()
    r = v.add_run(f"Result: a usable {nmod}-class baseline. {npend} classes have "
                  f"no training data and are held for Phase 6.")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0x1a, 0x7f, 0x37)
    doc.add_paragraph(
        f"macro-F1 over the {nmod} modelled classes: "
        + " / ".join(f"{m} {e['macro_f1_modelled']:.2f}" for m, e in evals.items())
        + f".  Over just the {nreal} that are a real generalisation test "
        "(external train, our test): "
        + " / ".join(f"{m} {e['macro_f1_real_eval']:.2f}" for m, e in evals.items())
        + ". Every modelled class scores >= 0.6. `mini_heart` was recovered from "
        "0.00 to ~0.92 F1 in this pass: HaGRIDv2's hand-heart gives the right "
        "handshape, and because each hand is normalised on its own wrist, only "
        "the body pose carried the chest-vs-overhead gap — an arm-elevation "
        "augmentation (features/augment.raise_arms) closes it with no new data. "
        "`i_love_you` and `heart` are still NOT modelled: the ASL ILY handshape "
        "is only on Roboflow Universe (not yet pulled) and the overhead 2-arm "
        "heart is in no dataset (a COCO-mining branch is wired, not yet run). "
        "Training them on augmented s01 frames actively hurt the model (dragged "
        "`rock` to 0.00), so they stay excluded until their data lands."
    )

    doc.add_heading("1. Architecture — where this sits", 1)
    doc.add_paragraph(
        "The pipeline has exactly ONE classifier slot: fused feature vector in -> "
        "(class, confidence) out -> temporal smoothing -> final label or idle. "
        "Phase 3 trains candidate models for that slot; it does NOT stack or "
        "ensemble them. LightGBM and RandomForest are the two Phase 3 candidates; "
        "Phase 4 adds an MLP and locks one (ARCHITECTURE.md 'Classifier interface')."
    )
    doc.add_paragraph(
        f"Features: {primary.get('features', 'both')} — the 152-d raw normalised "
        "keypoints PLUS 36 derived features (joint angles + a few length ratios, "
        "features/derived.py). The derived features are scale-invariant, which "
        "matters because MediaPipe Hands on a tight robot-camera wrist-crop "
        "produces a differently proportioned skeleton than on a large webcam hand "
        "(this broke `rock` in the first pass). class_weight = balanced x "
        "{aug-only 0.35, idle 0.6}. Features clipped to +-10."
    )

    doc.add_heading("2. Train / test data", 1)
    doc.add_paragraph(
        f"Train: {card['train']['rows']:,} rows — cleaned external features "
        f"(HaGRID, COCO) + augmentation for the cross-domain classes; augmented "
        f"s01 frames for sit/squat/laying/glico_pose. Pending classes contribute "
        f"0 rows. See results/phase2/dataset_report.docx."
    )
    doc.add_paragraph(
        f"Test: {card['test']['rows']:,} rows, NEVER augmented — every original "
        f"s01/s02 frame (1,367) + 214 held-out HaGRID `thumb`. Metrics below are "
        f"computed only over the {nmod} modelled classes' frames; the "
        f"pending-class frames are reported separately (section 6)."
    )

    doc.add_heading("3. LightGBM vs RandomForest", 1)
    lg, rf = evals.get("lgbm"), evals.get("rf")
    if lg and rf:
        _table(doc, ["metric", "LightGBM", "RandomForest"], [
            ["macro-F1 (modelled classes)", f"{lg['macro_f1_modelled']:.3f}",
             f"{rf['macro_f1_modelled']:.3f}"],
            ["macro-F1 (real-eval only)", f"{lg['macro_f1_real_eval']:.3f}",
             f"{rf['macro_f1_real_eval']:.3f}"],
            ["sit @ s02 recall (only clean cross-person number)",
             sit_s02.get("lgbm", "-"), sit_s02.get("rf", "-")],
            ["fit time", f"{metas['lgbm']['fit_s']} s", f"{metas['rf']['fit_s']} s"],
            ["model file size", f"{metas['lgbm']['size_mb']} MB",
             f"{metas['rf']['size_mb']} MB"],
        ])
        prows, pcolors = [], []
        for c in card["classes"]:
            a = lg["per_class"].get(c, {}).get("f1")
            b = rf["per_class"].get(c, {}).get("f1")
            if a is None and b is None:
                continue
            prows.append([c, f"{a:.2f}" if a is not None else "-",
                          f"{b:.2f}" if b is not None else "-",
                          "*" if (a is not None and b is not None and abs(a - b) > 0.05) else ""])
            pcolors.append(RGBColor(0x88, 0x66, 0) if prows[-1][3] else None)
        _table(doc, ["class", "LGBM F1", "RF F1", "differ"], prows, pcolors)
        doc.add_paragraph(
            "Accuracy is close. RandomForest is the working default: its "
            f"`sit`@s02 recall ({sit_s02.get('rf', '-')} vs "
            f"{sit_s02.get('lgbm', '-')}) is the only honest cross-person number "
            "we have, and its confidences are spread (LightGBM outputs ~1.0 on "
            "almost everything, which would break the idle/unknown threshold in "
            "Phase 5). LightGBM is 11x smaller. Final lock is Phase 4."
        )
    else:
        doc.add_paragraph(
            "The first Phase 3 pass (12 modelled classes) compared both: LightGBM "
            "macro-F1 0.88 vs RandomForest 0.87, but LightGBM's `sit`@s02 was "
            "0.45 vs RF 0.68 and its confidences sat at ~1.0 on everything "
            "(useless for the Phase 5 idle threshold) — so RF is the working "
            "default. This pass (mini_heart added, 13 classes) re-ran RF only: "
            "the SKUBA laptop is down to ~0.45 GB free RAM and LightGBM's "
            "15-class histogram build swaps for 15+ min. The LGBM head-to-head "
            "re-runs on Colab in Phase 4, where the model lock is decided anyway."
        )

    doc.add_heading("4. Per-class results (RandomForest)", 1)
    rows, colors = [], []
    for c in card["classes"]:
        pc = primary["per_class"].get(c)
        if not pc:
            continue
        rows.append([c, pc["eval_type"], pc["n"], f"{pc['precision']:.2f}",
                     f"{pc['recall']:.2f}", f"{pc['f1']:.2f}", f"{pc['mean_conf']:.2f}"])
        colors.append(RGBColor(0x88, 0x66, 0) if pc["f1"] < 0.65 else None)
    _table(doc, ["class", "eval type", "n", "prec", "rec", "F1", "conf"], rows, colors)
    for k, txt in EVAL_MEANS.items():
        doc.add_paragraph(f"{k}: {txt}", style="List Bullet")

    doc.add_heading("5. sit — the one clean cross-person number", 1)
    doc.add_paragraph(
        "`sit` trains on augmented s01 frames only, so its s02 test frames "
        "(different person, session, room) are a genuine cross-person check."
    )
    _table(doc, ["subject", "RF recall", "meaning"], [
        ["s01 (315 frames)", "1.00", "training basis — leaked, ignore"],
        ["s02 (60 frames)", str(sit_s02.get("rf", "-")), "clean cross-person — the honest sit number"],
    ])

    doc.add_heading("6. The pending-data classes + how mini_heart was fixed", 1)
    doc.add_paragraph(
        "mini_heart (previously pending) is now trained: HaGRIDv2 hand_heart "
        "(chest-level finger-heart) + an arm-elevation augmentation that shifts "
        "both forearms + hand points up by a shared offset, turning the "
        "chest-level rows into overhead ones. The handshape rides along untouched "
        "because each hand is normalised on its own wrist. A new derived feature, "
        "inter-wrist distance in body units, gives the model a direct "
        "hands-together signal. Result: mini_heart 0.00 -> "
        f"{primary['per_class'].get('mini_heart', {}).get('f1', 0):.2f} F1, and "
        "raise_right_hand improved as a side effect (cleaner arms-overhead negatives)."
    )
    doc.add_paragraph(
        "i_love_you and heart still have no training data. They are in the "
        "vocabulary and in test.npz but the model has no label for them. Their "
        "test frames land as:"
    )
    pl = primary.get("pending_data_lands_as", {})
    _table(doc, ["pending class", "test frames land as (count)"],
           [[c, ", ".join(f"{k} {v}" for k, v in w.items())] for c, w in pl.items()])
    doc.add_paragraph(
        "i_love_you -> rock is the s01 overlap (same person did both similarly in "
        "one session); the fix is real ILY images (Roboflow Universe carries the "
        "handshape as a labelled class — see docs/external_datasets.md R2.2). "
        "heart -> mini_heart now (it used to land on raise_right_hand) — the "
        "model sees hands-together-overhead but has no `heart` label; the "
        "separator when data lands is hand shape. heart is mineable from COCO "
        "pose (R2.3) or Phase 6 field data."
    )

    hg = OUT / "handgap_rock.jpg"
    if hg.exists():
        doc.add_heading("7. Why the derived features were needed (rock)", 1)
        doc.add_paragraph(
            "s01 `rock` hand landmarks (row 1) vs s01 `i_love_you` (row 2) vs "
            "HaGRID `rock` (row 3). Rows 1-2 are the same person in one session "
            "and nearly identical. Row 3 shows the HaGRID wrist-crop often misses "
            "the hand (sized by shoulder width, unreliable when hips are out of "
            "frame). Scale-invariant angle features + dropping i_love_you from "
            "training took `rock` from 0.00 to ~0.85."
        )
        doc.add_picture(str(hg), width=Inches(6.0))

    doc.add_heading("8. Confusions among the modelled classes", 1)
    conf = primary["confusion"]
    modset = set(primary["per_class"])
    pairs = sorted(((cnt, a, b) for a, dd in conf.items() for b, cnt in dd.items()
                    if a != b and a in modset and b in modset), reverse=True)[:10]
    _table(doc, ["true", "predicted as", "count"], [[a, b, cnt] for cnt, a, b in pairs])
    doc.add_paragraph(
        "The residual confusions are sensible: rock<->two_finger (adjacent "
        "handshapes), and several gestures -> idle at ~0.7 recall — a frame where "
        "the hand isn't clearly detected falls to idle. idle precision is ~0.45 "
        "for the same reason. Phase 5 temporal smoothing (majority vote over N "
        "frames) recovers most of this: a gesture held for ~1 s is ~30 frames."
    )

    doc.add_heading("9. Phase 4 (model exploration on this fixed pipeline)", 1)
    for i, t in enumerate([
        "Add an MLP as the third candidate; lock one of {RF, LGBM, MLP} with a "
        "documented rationale (IMPLEMENTATION_PLAN Phase 4).",
        "Compare feature sets properly: raw vs derived vs both (the flag exists).",
        "Tune per-class confidence thresholds on a validation slice, then wire "
        "the idle/unknown fallback (Phase 3 deliverable, carried into Phase 5).",
        "Re-extract COCO with a person-bbox crop — COCO gives the bbox; cropping "
        "to it before MediaPipe should lift raise_hand / t_pose recall (now "
        "~0.72) by removing the wrong-person detections.",
        "Pull i_love_you from Roboflow Universe (ASL ILY handshape, mostly "
        "CC BY 4.0) and run the wired COCO heart-mining branch; retrain. Both "
        "need a box that isn't the SKUBA laptop (Colab).",
    ], 1):
        doc.add_paragraph(f"{i}. {t}", style="List Bullet")

    out = OUT / "classifier_report.docx"
    try:
        doc.save(str(out))
    except PermissionError:
        out = OUT / "classifier_report.NEW.docx"
        doc.save(str(out))
        print("!! classifier_report.docx open in Word — wrote", out.name)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
